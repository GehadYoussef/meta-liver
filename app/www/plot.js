// Shiny output binding for figures built by plot_ly() in R/plot.R
(function() {
  var binding = new Shiny.OutputBinding();
  $.extend(binding, {
    find: function(scope) { return $(scope).find('.mash-plot'); },
    renderValue: function(el, fig) {
      if (!fig) { Plotly.purge(el); return; }
      var layout = Array.isArray(fig.layout) ? {} : fig.layout;
      var config = Object.assign({responsive: true, displaylogo: false}, Array.isArray(fig.config) ? {} : fig.config);
      if (!el.querySelector('.plot-container')) {
        el.innerHTML = '';
        Plotly.newPlot(el, fig.data, layout, config).then(function() {
          el.on('plotly_click', function(e) {
            var p = e.points[0];
            Shiny.setInputValue(el.id + '_click', {x: p.x, y: p.y}, {priority: 'event'});
          });
        });
      } else {
        Plotly.react(el, fig.data, layout, config);
      }
    },
    renderError: function(el, err) {
      Plotly.purge(el);
      el.innerHTML = err.message ? '<div class="plot-msg">' + $('<div>').text(err.message).html() + '</div>' : '';
    },
    clearError: function(el) {}
  });
  Shiny.outputBindings.register(binding, 'mash.plot');
})();
