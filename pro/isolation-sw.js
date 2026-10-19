/* Header shim for clients of /pro/. No response storage or offline behavior. */
self.addEventListener('install',event=>event.waitUntil(self.skipWaiting()));
self.addEventListener('activate',event=>event.waitUntil(self.clients.claim()));
self.addEventListener('fetch',event=>{
  const request=event.request,url=new URL(request.url);
  if(url.origin!==self.location.origin)return;
  if(request.cache==='only-if-cached'&&request.mode!=='same-origin')return;
  event.respondWith((async()=>{
    const response=await fetch(request);
    if(response.status===0||response.type==='opaque'||response.type==='opaqueredirect')return response;
    if(response.url&&new URL(response.url).origin!==self.location.origin)return response;
    const headers=new Headers(response.headers);
    headers.set('Cross-Origin-Opener-Policy','same-origin');
    headers.set('Cross-Origin-Embedder-Policy','require-corp');
    return new Response(response.body,{status:response.status,statusText:response.statusText,headers});
  })());
});
