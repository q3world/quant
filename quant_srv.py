#!/usr/bin/env python3

import os
import json
from typing import Union

from aiohttp import web

WS_FILE = os.path.join(os.path.dirname(__file__), 'quant.html')

c_client = []
p_client = []

async def wshandler(request: web.Request) -> Union[web.WebSocketResponse, web.Response]:
    response = web.WebSocketResponse()
    available = response.can_prepare(request)
    if not available:
        with open(WS_FILE, 'rb') as fp:
            return web.Response(body=fp.read(), content_type='text/html')

    await response.prepare(request)

    print(response)
    
    await response.send_str('Welcome!!!')

    try:
        print('Someone joined.')
        for ws in request.app['sockets']:
            await ws.send_str('Someone joined')
            
        request.app['sockets'].append(response)

        async for msg in response:
            if msg.type == web.WSMsgType.TEXT:
                print(msg.data)
                root = json.loads(msg.data)
                if 'pid' in root:
                    if not response in c_client:
                        c_client.append(response)

                elif 'key' in root:
                    if not response in p_client:
                        p_client.append(response)

                elif 'data' in root:
                    for ws in c_client:
                        await ws.send_str(msg.data)
                        
            else:
                return response
            
        return response

    finally:
        if response in c_client:
            c_client.remove(response)
            
        if response in p_client:
            p_client.remove(response)
            
        request.app['sockets'].remove(response)
        print('Someone disconnected.')
        for ws in request.app['sockets']:
            await ws.send_str('Someone disconnected.')


async def on_shutdown(app: web.Application) -> None:
    for ws in app['sockets']:
        await ws.close()


def init() -> web.Application:
    app = web.Application()
    app['sockets'] = []
    app.router.add_get('/', wshandler)
    app.on_shutdown.append(on_shutdown)
    return app


web.run_app(init(), port=5050)
