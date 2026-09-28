import pathlib
path = pathlib.Path('src/web/index.html')
content = path.read_text('utf-8')
old_str = "<input type=\"text\" class=\"search-input\" id=\"searchInput\" placeholder=\"Visual search (e.g. 'car', 'sunset') or in-photo text in quotes (e.g. '&quot;shawaya&quot;', '&quot;invoice&quot;')...\" onkeydown=\"if(event.key==='Enter') executeSearch()\">"
new_str = "<input type=\"text\" class=\"search-input\" id=\"searchInput\" placeholder=\"Visual search (e.g. 'car', 'sunset') or in-photo text in quotes (e.g. '&quot;shawaya&quot;', '&quot;invoice&quot;')...\" oninput=\"if(!this.value.trim()) activePersonId = null;\" onkeydown=\"if(event.key==='Enter') executeSearch()\">"

if old_str in content:
    content = content.replace(old_str, new_str)
    path.write_text(content, 'utf-8')
    print('Patched successfully!')
else:
    print('String not found!')
