fetch('/admin/stats').then(r=>r.json()).then(d=>{document.getElementById('out').textContent=JSON.stringify(d,null,2)});
