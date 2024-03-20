<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script>
;$('.tsd-page-title').html('');
;$('h4').filter(function () {
  return $(this).text() === 'Description';
}).html('');
;$(document).ready(()=>{
;$('a.tsd-index-link').filter(function(){
  return $(this).attr('href')=='modules/_internals_.html';
}).closest('section').html('');
});
</script>