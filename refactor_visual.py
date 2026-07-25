import re

filepath = r'C:\Users\alvar\OneDrive\Documentos\Proyectos-Personales\PROYECTO_FIA\visual.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all ft.colors.* to ft.Colors.*
content = re.sub(r'ft\.colors\.', 'ft.Colors.', content)

# Replace all ft.icons.* to ft.Icons.*
content = re.sub(r'ft\.icons\.', 'ft.Icons.', content)

# Replace all ft.alignment.* to ft.Alignment.*
content = re.sub(r'ft\.alignment\.center', 'ft.Alignment.CENTER', content)
content = re.sub(r'ft\.alignment\.top_center', 'ft.Alignment.TOP_CENTER', content)

# Replace ImageFit
content = re.sub(r'ft\.ImageFit\.', 'ft.BoxFit.', content)

# Replace page attributes
content = re.sub(r'page\.window_width', 'page.width', content)
content = re.sub(r'page\.window_height', 'page.height', content)

# Replace border.all
content = re.sub(r'ft\.border\.all', 'ft.Border.all', content)

# ElevatedButton
content = re.sub(r'ft\.ElevatedButton\(text=', 'ft.Button(', content)

# Dropdown on_change -> on_select
content = re.sub(r'(ft\.Dropdown\([^)]*)on_change=', r'\1on_select=', content)

# Tabs restructure
old_tabs = '''    tabs=ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(text="Modelo K-NN", icon=ft.Icons.IMAGE_ROUNDED, content=contenido_linea),
            ft.Tab(text="Árboles de decisión", icon=ft.Icons.BAR_CHART, content=contenido_linea_tree)
        ],
        expand=1,
        label_color=ft.Colors.BLUE_500,
        indicator_color=ft.Colors.BLUE_500,
        unselected_label_color=ft.Colors.WHITE
        
    )

    page.add(tabs)'''

new_tabs = '''    content_container = ft.Container(content=contenido_linea, expand=True)

    def tab_changed(e):
        if e.control.selected_index == 0:
            content_container.content = contenido_linea
        else:
            content_container.content = contenido_linea_tree
        page.update()

    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        on_change=tab_changed,
        tabs=[
            ft.Tab(label="Modelo K-NN", icon=ft.Icons.IMAGE_ROUNDED),
            ft.Tab(label="Árboles de decisión", icon=ft.Icons.BAR_CHART)
        ],
        expand=False,
        label_color=ft.Colors.BLUE_500,
        indicator_color=ft.Colors.BLUE_500,
        unselected_label_color=ft.Colors.WHITE
    )

    page.add(ft.Column([tabs, content_container], expand=True))'''

content = content.replace(old_tabs, new_tabs)

# ft.app -> ft.app
content = re.sub(r'ft\.app\(main\)', 'ft.run(main)', content)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done!')
