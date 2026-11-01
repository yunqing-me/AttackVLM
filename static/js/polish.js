// Add figure enlargement without rearranging any original page content.
const dialog = document.createElement('dialog');
dialog.className = 'figure-dialog';
dialog.setAttribute('aria-labelledby', 'figure-caption');
dialog.innerHTML = '<div class="figure-toolbar"><p id="figure-caption"></p><div class="figure-controls"><button type="button" class="figure-zoom" aria-pressed="false">Original size</button><button type="button" class="figure-close" aria-label="Close figure">Close ×</button></div></div><div class="figure-viewport" role="region" aria-label="Figure viewer" tabindex="0"><img alt=""></div><div class="figure-footer"><p class="figure-hint" aria-live="polite"></p><a class="figure-original" target="_blank" rel="noopener">Open full-resolution image ↗</a></div>';
document.body.append(dialog);
const figureViewport = dialog.querySelector('.figure-viewport');
const zoomButton = dialog.querySelector('.figure-zoom');

function setFigureZoom(originalSize) {
  figureViewport.dataset.originalSize = String(originalSize);
  zoomButton.setAttribute('aria-pressed', String(originalSize));
  zoomButton.textContent = originalSize ? 'Fit to window' : 'Original size';
  dialog.querySelector('.figure-hint').textContent = originalSize ? 'Scroll to inspect the full-resolution figure.' : 'Select Original size to inspect labels and details.';
  figureViewport.scrollLeft = 0;
  figureViewport.scrollTop = 0;
}

zoomButton.addEventListener('click', () => {
  setFigureZoom(figureViewport.dataset.originalSize !== 'true');
});

document.querySelectorAll('.section img.round').forEach(image => {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'paper-figure';
  button.title = 'Click to enlarge';
  button.setAttribute('aria-label', 'Enlarge figure: ' + image.alt);
  image.before(button);
  button.append(image);
  button.addEventListener('click', () => {
    const enlarged = figureViewport.querySelector('img');
    enlarged.src = image.src;
    enlarged.alt = image.alt;
    dialog.querySelector('#figure-caption').textContent = image.alt;
    dialog.querySelector('.figure-original').href = image.src;
    const originalSize = window.matchMedia('(max-width: 768px)').matches;
    setFigureZoom(originalSize);
    dialog.showModal();
    if (originalSize) figureViewport.focus();
  });
});

dialog.querySelector('.figure-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
});
