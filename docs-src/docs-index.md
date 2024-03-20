<script src="https://cdn.jsdelivr.net/npm/ndarray-js@1.0.0/dist/index.js"></script>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/dist/plot.umd.min.js"></script>
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.62.0/codemirror.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.62.0/codemirror.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.62.0/mode/javascript/javascript.min.js"></script>
<script>
  var __console_log = console.log;
  var __log_elem = null;
  console.log = function(...args){
    __console_log(...args);
    if (__log_elem !== null) {
      ;$(__log_elem).append(args.join(' ') + '\n');
    }
  }
</script>
<style>
  .tsd-page-toolbar{
    z-index: 5; /* top bar on top of codemirror code and gutters */
  }
  .hbox {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
  }
  .vbox {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  svg-outputs{
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    font-family: 'Courier New', Courier, monospace;
  }
  .justify-start {
    justify-content: flex-start;
  }
  /* Define keyframes for the gliding light effect */
@keyframes gliding-light {
  0% { box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.8); }
  50% { box-shadow: 150px 0px 200px rgba(255, 255, 255, 0.2); }
  100% { box-shadow: 300px 0px 300px rgba(255, 255, 255, 0); }
}
/* Apply the gliding light animation to the button */
.gliding-light-button {
  animation: gliding-light 3s;
}
/* Optional: You can add hover effect */
.gliding-light-button:hover {
  background-color: #0056b3; /* Change color on hover */
}
/* Apply the shading animation to the button */
 .execute-button {
  padding: 10px 20px;
  border: none;
  background-color: #007bff; /* Change this to your desired button color */
  color: white;
  font-size: 16px;
  cursor: pointer;
  }
 /* Optional: You can add hover effect */
 .execute-button:hover {
  background-color: #0056b3; /* Change color on hover */
}
/* Define keyframes for the shading effect */
@keyframes shade {
  0% { box-shadow: 0px 0px 0px rgba(0, 0, 0, 0.5); }
  50% { box-shadow: 8px 0px 15px rgba(0, 0, 0, 0.5); }
  100% { box-shadow: 16px 0px 30px rgba(0, 0, 0, 0.5); }
}
.shaded-button {
  animation: shade 2s infinite alternate; /* Apply the shading animation */
}
</style>
<div class="vbox" style="width:100%">
<!-- 
<h3>Header:</h3>
<div style="border: solid 1px black; height: fit-content;">
  <textarea id="codeHeader" disabled="true" style="width: 90vw; height:6.5em"></textarea>
  <script>
    (()=>{
      let scripts = [
        "https://d3js.org/d3.v7.min.js",
        "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/dist/plot.umd.min.js",
        "https://cdn.jsdelivr.net/npm/ndarray-js@1.0.0/dist/index.js",
      ].map(s=>'<' + 'script src="' + s + '"' + '>'+'<'+'/'+'script'+'>').join('\n');
      ;$('#codeHeader').text(scripts);
    })();
  </script>
</div> -->
<div style="border: solid 1px black;">
<textarea id="codeInput">
  </textarea>
</div>
<div class="hbox" style="width:100%">
  <div></div>
  <button class="execute-button" style="padding:1em; margin:1em; font-size: large;">Execute</button>
  <div></div>
</div>
</div>
</div>
<div class="vbox justify-start">
<div class="vbox">
  <h4>Errors:</h4>
  <textarea disabled="true" id="stderr" cols="50" style="border:none; min-height: 20vh;">
    (error logs will appear here)
  </textarea>
</div>
<div class="vbox">
  <h4>Console output:</h4>
  <textarea disabled="true" id="stdout" cols="50" style="border:none; min-height: 20vh;">
    (console.log logs will appear here)
  </textarea>
  <h4>SVG outputs:</h4>
  <div id="svg-outputs">
  </div>
</div>
</div>
<script>
  ;$('#stderr').parent().hide();
  ;$('#stdout').parent().hide();
  ;$('.tsd-panel-group.tsd-index-group').html('');
  ;$(document).ready(()=>{
    ;$('.tsd-panel-group.tsd-index-group').html('');
    let initialCode = `
// Part 1: data creation
// (ENTER)
var XY = np.random.randn([5000, 2])
var norm = np.norm(XY, { axis: -1, keepdims: true });
console.log(np.allclose(norm, XY.pow(2).sum(-1).index('...', 'None').pow(0.5)));
var XY_unit = XY.op('/', norm);
var angle = 45; // <-- rotate me
var group = np.atan2(XY.index(':', 1), XY.index(':', 0)).multiply(180/np.pi).add(90-angle).abs().greater(90);
// (ENTER)
console.log(\`\\nFirst five points:\\n\${XY.index(\`0:5\`)}\`);
console.log(\`\\nNorm of the first five points (before and after):\`);
console.log(np.stack([norm.index(':', 0), XY_unit.norm(-1)], -1).index(\`0:5\`));
// (ENTER)
// Part 2: plots
var svg = Plot.plot({
grid: true,
color: {scheme: "Observable10"},
aspectRatio: 1, // undefined,
marks: [
  Plot.dot(XY.tolist().map(([x,y],i)=>({x, y, group:group.index(i)})), {x: "x", y: "y", r:1, stroke:"group"}),
  Plot.dot(XY_unit.tolist().map(([x,y])=>({x, y})), {x: "x", y: "y", r:1, fill:"#b36969"}),
]
});
document.querySelector('#svg-outputs').append(svg);
// (ENTER)
var svg = Plot.plot({
grid: true,
color: {scheme: "Observable10"},
marks: [
  Plot.rectY(
    XY.tolist().map(([x,y], i)=>({x, y, group:group.index(i)})), Plot.binX({y2: "count"},
    {x: "x", fill:"group", mixBlendMode: "screen"})
  ),
]
});
document.querySelector('#svg-outputs').append(svg);
// (ENTER)
var x = np.linspace(-5, 5, 500)
// alt: var y = np.exp(x.pow(2).negative().divide(2)).multiply(x.shape[0]/np.sqrt(2*np.pi));
var y = np\`np.exp(-\${x}**2 / 2) * \${x.shape[0]/np.sqrt(2*np.pi)}\`;
// (ENTER)
var svg = Plot.plot({
grid: true,
color: {scheme: "Observable10"},
marks: [
  Plot.rectY(
    XY.tolist().map(([x,y], i)=>({x, y, group:group.index(i)})),
    Plot.binX({y: "count"}, {x: "x", fill:"group", mixBlendMode: "screen"}),
  ),
  Plot.dot(np.stack([x, y], axis=-1).tolist().map(([x,y], i)=>({x, y})), {x: "x", y: "y", r:1}),
]
});
document.querySelector('#svg-outputs').append(svg);`.replace(new RegExp('// \\(ENTER\\)', 'g'), '');
    var codeInput = document.getElementById("codeInput");
    // ;$('#codeInput').text(initialCode);
    var codeEditor = CodeMirror.fromTextArea(codeInput, {
      mode: "javascript",
      lineNumbers: true,
      theme: "default",
      autoRefresh:true,
      autoCloseBrackets: true,
      matchBrackets: true,
      height: '60vh', 
    });
    codeEditor.setValue(initialCode);
    codeEditor.refresh();
    setTimeout(function() {codeEditor.refresh();}, 200);
    // setTimeout(()=>$('div.CodeMirror-scroll').trigger('click'), 1000);
    let first=true;
    setTimeout(()=>$('.execute-button').addClass('gliding-light-button'), 1500);
    setTimeout(()=>$('.execute-button').addClass('shaded-button'), 1500+5000);
    ;$('.execute-button').on('click', ()=>{
      ;$('.execute-button').removeClass('shaded-button');
      var code = codeEditor.getValue();
      ;$('#stdout').parent().show();
      ;$('#svg-outputs').html('');
      ;$('#stdout').html('');
      ;$('#stderr').html('');
      ;$('#stderr').parent().hide();
      ;$('#stderr').parent().hide();
      __log_elem = $('#stdout');
      try{
        eval(code);
      } catch(e){
        ;$('#stderr').parent().show();
        ;$('#stderr').append(e.stack)
        ;$('#stderr').append('Press F12 for more details')
        throw e;
      }
      if(first){
      window.scrollBy({
        top: 500,
        behavior: 'smooth' // Smooth scrolling animation
      });
      }
      first=false;
    });
  });
</script>