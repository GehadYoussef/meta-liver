// Shiny output binding for tables built by rt() in R/table.R
(function() {
  function esc(s) { return String(s).replace(/[&<>"]/g, function(c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }

  function draw(el) {
    var t = el._table, s = el._state, n = t.text.length;
    var rows = [];
    for (var i = 0; i < n; i++) if (!s.query || t.text[i].indexOf(s.query) >= 0) rows.push(i);
    if (s.sort !== null) {
      var key = t.cols[s.sort].sort, dir = s.dir;
      rows.sort(function(a, b) {
        var x = key[a], y = key[b];
        if (x === null || x === '') return 1;
        if (y === null || y === '') return -1;
        return (x < y ? -1 : x > y ? 1 : 0) * dir;
      });
    }
    var pages = Math.max(1, Math.ceil(rows.length / t.page));
    s.p = Math.min(s.p, pages - 1);
    var shown = rows.slice(s.p * t.page, (s.p + 1) * t.page);
    var head = t.cols.map(function(c, j) {
      var mark = s.sort === j ? (s.dir > 0 ? ' ▴' : ' ▾') : '';
      return '<th data-col="' + j + '" style="text-align:' + c.align + ';' + c.style + '">' + esc(c.name) + mark + '</th>';
    }).join('');
    var body = shown.map(function(i) {
      return '<tr>' + t.cols.map(function(c) {
        return '<td style="text-align:' + c.align + ';' + c.style + '">' + c.cells[i] + '</td>';
      }).join('') + '</tr>';
    }).join('');
    var from = rows.length ? s.p * t.page + 1 : 0, to = s.p * t.page + shown.length;
    el.querySelector('.mt-body').innerHTML =
      '<div class="mt-scroll"><table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>' +
      (rows.length ? '' : '<div class="mt-empty">No rows found</div>') +
      (rows.length > t.page || s.query ? '<div class="mt-foot"><span>' + from + '–' + to + ' of ' + rows.length + ' rows</span>' +
        '<span class="mt-pager"><button data-step="-1"' + (s.p === 0 ? ' disabled' : '') + '>Previous</button>' +
        '<span>' + (s.p + 1) + ' of ' + pages + '</span>' +
        '<button data-step="1"' + (s.p >= pages - 1 ? ' disabled' : '') + '>Next</button></span></div>' : '');
  }

  var binding = new Shiny.OutputBinding();
  $.extend(binding, {
    find: function(scope) { return $(scope).find('.mash-table'); },
    renderValue: function(el, t) {
      if (!t) { el.innerHTML = ''; return; }
      el._table = t;
      el._state = {query: '', sort: null, dir: 1, p: 0};
      el.innerHTML = (t.searchable ? '<div class="mt-top"><input type="search" class="mt-search" placeholder="Search"></div>' : '') +
                     '<div class="mt-body"></div>';
      draw(el);
      if (el._wired) return;
      el._wired = true;
      el.addEventListener('input', function(e) {
        if (!e.target.classList.contains('mt-search')) return;
        el._state.query = e.target.value.trim().toLowerCase();
        el._state.p = 0;
        draw(el);
      });
      el.addEventListener('click', function(e) {
        var th = e.target.closest('th[data-col]'), btn = e.target.closest('button[data-step]');
        var s = el._state;
        if (th) {
          var j = +th.dataset.col;
          if (s.sort === j) s.dir = -s.dir; else { s.sort = j; s.dir = 1; }
          draw(el);
        } else if (btn && !btn.disabled) {
          s.p += +btn.dataset.step;
          draw(el);
        }
      });
    },
    renderError: function(el, err) {
      el.innerHTML = err.message ? '<div class="plot-msg">' + esc(err.message) + '</div>' : '';
    },
    clearError: function(el) {}
  });
  Shiny.outputBindings.register(binding, 'mash.table');
})();
