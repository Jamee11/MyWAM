"""Bounded request aggregation over one persistent WS link; per-session state stays separate.

relay runs on the model host against independent existing policy ports;
gateway runs on the simulator host and exposes ordinary policy ports.
No tensor batching, prefetch, replay, image recoding, or shared action buffers.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

ROOT=Path('/remote-home/jinminghao/OpenWAM')
sys.path.insert(0,str(ROOT))


async def relay(args):
    import websockets
    from benchmarks.utils.transport import WSPolicyClient
    clients=[WSPolicyClient(f'ws://127.0.0.1:{port}',timeout=120) for port in args.backend_ports]
    # Refuse to start against an old/non-chunk model or an unready endpoint.
    for client in clients:
        health=client.ping()
        assert health.get('type')=='pong'
        if not args.compat32:
            assert health.get('executable_action_chunk_v1')==dict(horizon=32)
    owner=asyncio.Lock()
    pool=ThreadPoolExecutor(max_workers=1)
    stats=dict(batches=0,requests=0,batch_sizes=[],model_forward_seconds=0.0,
               model_queue_wait_seconds=0.0,compat32=args.compat32)
    async def persist():
        while True:
            await asyncio.sleep(10)
            args.stats.write_text(json.dumps(stats,indent=2)+'\n')

    def forward(request):
        slot=request['slot']
        if not isinstance(slot,int) or not 0<=slot<len(clients):
            raise ValueError('Invalid session')
        payload=request['message'];op=payload.get('type','obs')
        if op not in ('obs','reset','ping'):raise ValueError('Invalid operation')
        begin=time.monotonic()
        # Even reset is not replayed after an ambiguous connection drop.
        key='executable_action_chunk_v1'
        if args.compat32 and op=='obs' and key in payload:
            if payload[key]!=dict(horizon=32):raise ValueError('Unsupported horizon')
            plain={k:v for k,v in payload.items() if k!=key}
            first=clients[slot]._roundtrip(plain,reconnect=False)
            if (first['step']-1)%32:raise ValueError('Backend buffer boundary mismatch; reset required')
            actions=[first['action']]
            for index in range(1,32):
                response=clients[slot]._roundtrip(plain,reconnect=False)
                if response['step']!=first['step']+index:raise ValueError('Backend sequence mismatch')
                actions.append(response['action'])
            result={**first,key:dict(actions=actions,start_step=first['step'],count=32),
                    'latency_ms':(time.monotonic()-begin)*1000,
                    'compatibility_legacy_backend':True}
        else:
            result=clients[slot]._roundtrip(payload,reconnect=False)
        if args.compat32 and op=='ping':
            result={**result,key:dict(horizon=32),'compatibility_legacy_backend':True}
        return dict(id=request['id'],result=result,model_forward_seconds=time.monotonic()-begin)

    async def handler(ws):
        if owner.locked():
            await ws.close(code=1008,reason='One gateway owns these sessions');return
        async with owner:
            queue=asyncio.Queue(maxsize=4)
            async def worker():
                while True:
                    batch,received=await queue.get();replies=[]
                    stats['batches']+=1;stats['batch_sizes'].append(len(batch))
                    stats['batch_sizes']=stats['batch_sizes'][-1000:]
                    for request in batch:
                        stats['model_queue_wait_seconds']+=time.monotonic()-received
                        try:reply=await asyncio.get_running_loop().run_in_executor(pool,forward,request)
                        except Exception as error:
                            reply=dict(id=request.get('id'),error=type(error).__name__+': '+str(error))
                        stats['requests']+=1
                        stats['model_forward_seconds']+=reply.get('model_forward_seconds',0)
                        if args.reply_mode=='stream':await ws.send(json.dumps(dict(replies=[reply])))
                        else:replies.append(reply)
                    if replies:await ws.send(json.dumps(dict(replies=replies)))
                    queue.task_done()
            task=asyncio.create_task(worker())
            try:
                async for message in ws:
                    batch=json.loads(message)['requests']
                    if not isinstance(batch,list) or not 1<=len(batch)<=args.max_batch:
                        raise ValueError('Invalid batch size')
                    await queue.put((batch,time.monotonic()))
            finally:
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)
                # Cancellation cannot stop a running inference thread. Wait for
                # its completion before permitting a new gateway to reset slots.
                await asyncio.get_running_loop().run_in_executor(pool,lambda:None)
                args.stats.write_text(json.dumps(stats,indent=2)+'\n')
    persistence=asyncio.create_task(persist())
    try:
        async with websockets.serve(handler,'127.0.0.1',args.port,max_size=16*1024*1024,
                                    compression=None,ping_interval=None):
            print('RELAY_READY',args.port,args.reply_mode,flush=True)
            await asyncio.Future()
    finally:
        persistence.cancel()
        await asyncio.gather(persistence,return_exceptions=True)
        pool.shutdown(wait=True)
        for client in clients:client.close()


async def gateway(args):
    import websockets
    queue=asyncio.Queue(maxsize=64);pending={};sequence=0
    stats=dict(requests=0,batches=0,batch_sizes=[],queue_wait_seconds=0.0)
    owners={}
    # Upstream loss invalidates the whole gateway; never replay unacknowledged obs.
    async with websockets.connect(f'ws://127.0.0.1:{args.port}',max_size=16*1024*1024,
                                  compression=None,ping_interval=None,proxy=None) as upstream:
        async def transmit():
            while True:
                item=await queue.get();batch=[item]
                if args.window_ms:await asyncio.sleep(args.window_ms/1000)
                while len(batch)<args.max_batch:
                    try:batch.append(queue.get_nowait())
                    except asyncio.QueueEmpty:break
                now=time.monotonic()
                stats['batches']+=1;stats['batch_sizes'].append(len(batch))
                stats['batch_sizes']=stats['batch_sizes'][-1000:]
                for row in batch:stats['queue_wait_seconds']+=now-row.pop('queued_at')
                await upstream.send(json.dumps(dict(requests=batch)))
                for _ in batch:queue.task_done()
        async def receive():
            async for raw in upstream:
                for reply in json.loads(raw)['replies']:
                    future=pending.pop(reply['id'])
                    if future.cancelled():continue
                    if 'error' in reply:future.set_exception(RuntimeError(reply['error']))
                    else:future.set_result(reply['result'])
            raise ConnectionError('Model relay disconnected; reset required')
        async def persist():
            while True:
                await asyncio.sleep(10)
                args.stats.write_text(json.dumps(stats,indent=2)+'\n')
        async def handler(ws,slot):
            nonlocal sequence
            try:
                async for raw in ws:
                    message=json.loads(raw);op=message.get('type','obs')
                    if op not in ('obs','reset','ping'):raise ValueError('Invalid operation')
                    if op in ('obs','reset'):
                        if slot in owners and owners[slot] is not ws:
                            raise ValueError('Session is owned by another simulator')
                        owners[slot]=ws
                    sequence+=1;key=sequence;future=asyncio.get_running_loop().create_future()
                    pending[key]=future;stats['requests']+=1
                    await queue.put(dict(id=key,slot=slot,message=message,queued_at=time.monotonic()))
                    result=await asyncio.wait_for(asyncio.shield(future),timeout=120)
                    await ws.send(json.dumps(result))
            except Exception as error:
                try:await ws.send(json.dumps(dict(type='error',code='internal_error',message=str(error))))
                except websockets.exceptions.ConnectionClosed:pass
                await ws.close()
            finally:
                if owners.get(slot) is ws:owners.pop(slot)
        servers=[]
        try:
            for slot,port in enumerate(args.listen_ports):
                async def lane(ws,slot=slot):return await handler(ws,slot)
                servers.append(await websockets.serve(lane,'127.0.0.1',port,
                    max_size=16*1024*1024,compression=None,ping_interval=None))
            print('GATEWAY_READY',args.listen_ports,'window_ms',args.window_ms,flush=True)
            tasks=[asyncio.create_task(f()) for f in (transmit,receive,persist)]
            done,remaining=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
            for future in pending.values():
                if not future.done():future.set_exception(ConnectionError('Gateway stopped; reset required'))
            for task in remaining:task.cancel()
            await asyncio.gather(*remaining,return_exceptions=True)
            for task in done:task.result()
        finally:
            for server in servers:server.close()
            await asyncio.gather(*(server.wait_closed() for server in servers))
            args.stats.write_text(json.dumps(stats,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role',choices=['relay','gateway'])
    p.add_argument('--port',type=int,default=34980)
    p.add_argument('--backend-ports',type=int,nargs='+',default=[34910,34911,34912])
    p.add_argument('--listen-ports',type=int,nargs='+',default=[35010,35011,35012])
    p.add_argument('--window-ms',type=float,default=2)
    p.add_argument('--max-batch',type=int,default=8)
    p.add_argument('--reply-mode',choices=['stream','grouped'],default='stream')
    p.add_argument('--compat32',action='store_true',help='Drain32 legacy backend requests locally, preserving live model processes')
    p.add_argument('--stats',type=Path,required=True)
    args=p.parse_args()
    if not 0<=args.window_ms<=10 or not 1<=args.max_batch<=8:p.error('Unbounded batching forbidden')
    asyncio.run(relay(args) if args.role=='relay' else gateway(args))


if __name__=='__main__':main()
