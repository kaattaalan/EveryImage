from pathlib import Path

html_path = Path("src/web/index.html")
content = html_path.read_text(encoding="utf-8")

# 1. Update search view HTML to add the filters panel
old_search_markup = '''    <div class="search-bar-wrap">
      <input type="text" class="search-input" id="searchInput" placeholder="Natural language search (e.g. 'Red car on beach', 'Al Gore speaking in a suit')..." onkeydown="if(event.key==='Enter') executeSearch()">
      <button class="btn btn-primary" onclick="executeSearch()">Search</button>
    </div>

    <div class="chips-row">'''

new_search_markup = '''    <div class="search-bar-wrap">
      <input type="text" class="search-input" id="searchInput" placeholder="Natural language search (e.g. 'Red car on beach', 'Al Gore speaking in a suit', or 'bus -blue')..." onkeydown="if(event.key==='Enter') executeSearch()">
      <button class="btn btn-primary" onclick="executeSearch()">Search</button>
      <button class="btn" id="btnToggleFilters" onclick="toggleSearchFilters()">Filters</button>
    </div>

    <!-- ADVANCED SEARCH FILTERS -->
    <div id="searchFiltersBox" style="background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; align-items: flex-end;">
        <div>
          <label style="font-size: 11px; color: var(--text-muted); display: block; margin-bottom: 4px;">Score Filter (Precision):</label>
          <select class="select-input" id="searchMinScore" style="width: 100%;" onchange="executeSearch()">
            <option value="0.04">High Precision (> 0.04)</option>
            <option value="0.00" selected>Balanced (> 0.00)</option>
            <option value="-0.05">Relaxed (> -0.05)</option>
            <option value="-1.0">Show All (No Filter)</option>
          </select>
        </div>

        <div>
          <label style="font-size: 11px; color: var(--text-muted); display: block; margin-bottom: 4px;">Negative Tags / Exclude:</label>
          <input type="text" class="select-input" id="searchNegativePrompt" placeholder="e.g. night, indoor, dog (or use -term)" style="width: 100%;" onkeydown="if(event.key==='Enter') executeSearch()">
        </div>

        <div>
          <label style="font-size: 11px; color: var(--text-muted); display: block; margin-bottom: 4px;">From Date:</label>
          <input type="date" class="select-input" id="searchFromDate" style="width: 100%;" onchange="executeSearch()">
        </div>

        <div>
          <label style="font-size: 11px; color: var(--text-muted); display: block; margin-bottom: 4px;">To Date:</label>
          <input type="date" class="select-input" id="searchToDate" style="width: 100%;" onchange="executeSearch()">
        </div>

        <div>
          <label style="font-size: 11px; color: var(--text-muted); display: block; margin-bottom: 4px;">Album Filter:</label>
          <select class="select-input" id="searchLibFilter" style="width: 100%;" onchange="executeSearch()"></select>
        </div>

        <div>
          <button class="btn" style="width: 100%;" onclick="resetSearchFilters()">Reset Filters</button>
        </div>
      </div>
    </div>

    <div class="chips-row">'''

content = content.replace(old_search_markup, new_search_markup)

# 2. Update executeSearch JS function to send filters
old_exec_search = '''async function executeSearch() {
  const query = document.getElementById('searchInput').value.trim();
  if (!query) return;

  const metaEl = document.getElementById('searchMeta');
  const gridEl = document.getElementById('searchResultsGrid');
  metaEl.textContent = 'Searching ONNX vector index...';
  gridEl.innerHTML = '';

  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const data = await res.json();'''

new_exec_search = '''function toggleSearchFilters() {
  const box = document.getElementById('searchFiltersBox');
  box.style.display = (box.style.display === 'none') ? 'block' : 'none';
}

function resetSearchFilters() {
  document.getElementById('searchMinScore').value = '0.00';
  document.getElementById('searchNegativePrompt').value = '';
  document.getElementById('searchFromDate').value = '';
  document.getElementById('searchToDate').value = '';
  document.getElementById('searchLibFilter').value = '';
  executeSearch();
}

async function executeSearch() {
  const query = document.getElementById('searchInput').value.trim();
  if (!query) return;

  const minScore = document.getElementById('searchMinScore').value;
  const negPrompt = document.getElementById('searchNegativePrompt').value.trim();
  const fromDate = document.getElementById('searchFromDate').value;
  const toDate = document.getElementById('searchToDate').value;
  const libId = document.getElementById('searchLibFilter').value;

  const metaEl = document.getElementById('searchMeta');
  const gridEl = document.getElementById('searchResultsGrid');
  metaEl.textContent = 'Searching ONNX vector index...';
  gridEl.innerHTML = '';

  let url = `/api/search?q=${encodeURIComponent(query)}&min_score=${minScore}`;
  if (negPrompt) url += `&negative_prompt=${encodeURIComponent(negPrompt)}`;
  if (fromDate) url += `&from_date=${fromDate}`;
  if (toDate) url += `&to_date=${toDate}`;
  if (libId) url += `&library_id=${libId}`;

  try {
    const res = await fetch(url);
    const data = await res.json();'''

content = content.replace(old_exec_search, new_exec_search)

# 3. Populate searchLibFilter in populateLibraryDropdowns
old_pop = '''    gSelect.innerHTML = opts;
    tSelect.innerHTML = opts;'''

new_pop = '''    gSelect.innerHTML = opts;
    tSelect.innerHTML = opts;
    const sSelect = document.getElementById('searchLibFilter');
    if (sSelect) sSelect.innerHTML = opts;'''

content = content.replace(old_pop, new_pop)

# 4. Update polling speed and dynamic progress text in updateScannerStatus
content = content.replace("pollInterval = setInterval(updateScannerStatus, 1000);", "pollInterval = setInterval(updateScannerStatus, 400);")

old_scanner_status_js = '''    document.getElementById('scanStatusTitle').textContent = `Engine: ${s.state.toUpperCase()}`;
    document.getElementById('scanCurrentFile').textContent = s.current_file || (isRunning ? 'Processing...' : 'Idle');'''

new_scanner_status_js = '''    let phaseTitle = `Engine: ${s.state.toUpperCase()}`;
    if (s.state === 'crawling') phaseTitle = 'Engine: DISCOVERING FILES';
    if (s.state === 'indexing') phaseTitle = 'Engine: ANALYZING & TAGGING (GPU)';
    document.getElementById('scanStatusTitle').textContent = phaseTitle;
    document.getElementById('scanCurrentFile').textContent = s.current_file || (isRunning ? 'Processing...' : 'Idle');'''

content = content.replace(old_scanner_status_js, new_scanner_status_js)

html_path.write_text(content, encoding="utf-8")
print("Updated src/web/index.html with search filters and dynamic progress bar!")
