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
<h3>Interactive demo (modify at will):</h3>
<div style="border: solid 1px black; height: fit-content;">
<textarea id="codeInput" style="height: fit-content !important;">
  </textarea>
  <script>
    (()=>{
      ;$('#codeInput').text(`
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
  Plot.rectY(XY.tolist().map(([x,y], i)=>({x, y, group:group.index(i)})), Plot.binX({y2: "count"}, {x: "x", fill:"group", mixBlendMode: "screen"})),
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
  Plot.rectY(XY.tolist().map(([x,y], i)=>({x, y, group:group.index(i)})), Plot.binX({y: "count"}, {x: "x", fill:"group", mixBlendMode: "screen"})),
  Plot.dot(np.stack([x, y], axis=-1).tolist().map(([x,y], i)=>({x, y})), {x: "x", y: "y", r:1}),
]
});
document.querySelector('#svg-outputs').append(svg);`.replace(new RegExp('// \\(ENTER\\)', 'g'), ''));
})()
    </script>
</div>
<div class="hbox" style="width:100%">
  <div></div>
  <button id="execute-button" style="padding:1em; margin:1em; font-size: large;">Execute</button>
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
  var codeInput = document.getElementById("codeInput");
  var codeEditor = CodeMirror.fromTextArea(codeInput, {
    mode: "javascript",
    lineNumbers: true,
    theme: "default",
    autoCloseBrackets: true,
    matchBrackets: true,
    height: '60vh', 
  });
  ;
  (()=>{
    let first=true;
    ;$('#execute-button').on('click', ()=>{
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
  })();
</script>