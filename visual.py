import flet as ft
# import main as m
import Funciones as f
import joblib
def set_size(e, page, container1, width_percentage1, height_percentage1, container2, width_percentage2, height_percentage2,
             container3, container4):
    container1.width = page.width * width_percentage1
    container1.height = page.height * height_percentage1
    container2.width = page.width * width_percentage2
    container2.height = page.height * height_percentage2

    container3.width = page.width * width_percentage1
    container3.height = page.height * height_percentage1
    container4.width = page.width * width_percentage2
    container4.height = page.height * height_percentage2
    page.update()

def change_image(page, contenedor, new_image_path: str):
        contenedor.content = ft.Image(
            src=new_image_path,  # Update the image path
            expand=True,
            fit=ft.BoxFit.CONTAIN,
        )
        page.update() 


def hacer_datos():
    data = f.get_data('games.csv')
    # Quitamos los valores atípicos del incrment code, deben 
    # Tener un porcentaje o un número mínimo
    data = f.filter(data, 'increment_code', 0.02)
    data = f.filter(data,'opening_eco', 0.02)
    data_2 = data.copy()
    
    return data,data_2


def main(page: ft.Page):
    page.title = "Containers - Clickable and Not"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.START
    page.theme_mode=ft.ThemeMode.LIGHT
    page.bgcolor=ft.Colors.BLUE_GREY_800
    model_tree = joblib.load('decision_tree_model.joblib')
    model_knn = joblib.load('k_nn_model.joblib')
    

    width_percentage_tablero=0.6
    height_percentage_tablero=0.9

    width_percentage_variables=0.3
    height_percentage_variables=0.9

    ruta_metricas_knn = "metrics_report_knn.txt"
    try:
        with open(ruta_metricas_knn, "r") as file:
            metricas_knn = file.read()
    except FileNotFoundError:
        metricas_knn = "Metrics file not found. Please run the model first."

    boton_matriz_confusion_knn = ft.IconButton(
        icon=ft.Icons.IMAGE,  # Icon for the button
        icon_size=30,
        tooltip="Matriz de confusión",
        on_click=lambda _: change_image(page, contenedor_imagen, "img2/matriz_confusion_knn.png"),  # New image path
    )

    boton_grafica_dispersion_knn = ft.IconButton(
        icon=ft.Icons.IMAGE,  # Icon for the button
        icon_size=30,
        tooltip="Grafica dispersion",
        on_click=lambda _: change_image(page, contenedor_imagen, "img2/Grafica_dispersion_knn.png"),  # New image path
    )

    user_data = {}

    # Function to update user data on change
    def update_data(e):
        # Update the user_data dictionary with the current values
        user_data['white_rating'] = white_rating_slider.value
        user_data['black_rating'] = black_rating_slider.value
        user_data['opening_ply'] = opening_ply_dropdown.value
        user_data['victory_status'] = victory_status_choices.value
        user_data['opening_eco'] = opening_eco_choices.value
        user_data['increment_code'] = increment_code_choices.value
        
        # Print the updated data (for testing purposes, could be saved or processed later)
        print("Knn")
        print(user_data)

        # Update the page (for immediate feedback)
        page.update()

    # Create sliders for the ratings
    white_rating_slider = ft.Slider(
        min=784,
        max=2622,
        divisions=2200,
        label="{value}",
        value=1500,
        on_change=update_data,  # Call the update_data function on change
    )

    black_rating_slider = ft.Slider(
        min=795,
        max=2723,
        divisions=2200,
        label="{value}",
        value=1500,
        on_change=update_data,  # Call the update_data function on change
    )

    # Create dropdown for opening play
    opening_ply_dropdown = ft.Dropdown(
        options=[ft.dropdown.Option(str(i)) for i in range(1, 14)],
        value="1",
        on_select=update_data,  # Call the update_data function on change
    )

    # Create RadioGroup for victory status (single choice)
    victory_status_choices = ft.RadioGroup(
        content=ft.Column([
            ft.Radio("Victory Status Mate", value=1),
            ft.Radio("Victory Status Resign", value=2),
            
        ]),
        value=1,  # Default value
        on_change=update_data,  # Call the update_data function on change
    )

    # Create RadioGroup for opening eco (single choice)
    opening_eco_choices = ft.RadioGroup(
        content=ft.Column([
            ft.Radio("Opening ECO B00", value=1),
            ft.Radio("Opening ECO B01", value=2),
            ft.Radio("Opening ECO C00", value=3),
            ft.Radio("Opening ECO C41", value=4),
            ft.Radio("Opening ECO D00", value=5),
            ft.Radio("Other", value=6),
        ]),
        value=1,  # Default value
        on_change=update_data,  # Call the update_data function on change
    )

    # Create RadioGroup for increment code (single choice)
    increment_code_choices = ft.RadioGroup(
        content=ft.Column([
            ft.Radio("Increment Code 10+0", value=1),
            ft.Radio("Increment Code 15+0", value=2),
            ft.Radio("Other", value=3),
        ]),
        value=1,  # Default value
        on_change=update_data,  # Call the update_data function on change
    )


    contenedor_imagen=ft.Container(
                    content=ft.Image(
                        src=f"img2/matriz_confusion_knn.png",
                        expand=True,
                        fit=ft.BoxFit.CONTAIN,
                    ),
                    margin=10,
                    padding=10,
                    alignment=ft.Alignment.CENTER,
                    bgcolor=ft.Colors.GREY,
                    expand=9,
                    border_radius=10,
    )


    eleccion_opciones=ft.Container(
                    content=ft.Row([boton_matriz_confusion_knn, boton_grafica_dispersion_knn],
                                   alignment=ft.Alignment.CENTER
                                   ),
                    alignment=ft.Alignment.TOP_CENTER,
                    bgcolor=ft.Colors.GREY,
                    expand=1,

    )
    
    columna_tablero=ft.Column([contenedor_imagen, eleccion_opciones])

    contenedor_metricas=ft.Container(
                    content=ft.Text(
                        value=metricas_knn,
                        size=14,  # Font size for better readability
                        selectable=True,  # Allow the user to select and copy text
                    ),
                    alignment=ft.Alignment.TOP_CENTER,
                    bgcolor=ft.Colors.GREY,
                    expand=1,
                    

    )

    data, data_2 = hacer_datos()
    
    contenedor_victoria_knn=ft.Container(
        content=ft.Text(value="Sin predecir"),
        padding=20,
        bgcolor="blue",
        border_radius=15,
        border=ft.Border.all(3, "white"),
        alignment=ft.Alignment.CENTER,
        )
    
    def on_button_click_knn(e):
        lista_knn = f.random_values_interactive( 
        data_2
        ,white_rating_slider.value
        ,black_rating_slider.value
        ,opening_ply_dropdown.value
        ,victory_status_choices.value
        ,opening_eco_choices.value
        ,increment_code_choices.value
        )
        prediccion = model_knn.predict(lista_knn)
        print(prediccion)

        if prediccion==1:
            contenedor_victoria_knn.content=ft.Text(value="Ganan las blancas", color=ft.Colors.BLACK)
            contenedor_victoria_knn.bgcolor=ft.Colors.WHITE
        else:
            
            contenedor_victoria_knn.content=ft.Text(value="Gana negro", color=ft.Colors.WHITE)
            contenedor_victoria_knn.bgcolor=ft.Colors.BLACK
        page.update()

    # Create a button and assign the click event
    boton_knn = ft.Button("Prediccion", on_click=on_button_click_knn)


    items=[
            ft.Text("Chess Game Input Form", size=20, weight=ft.FontWeight.BOLD),
            ft.Text("White Rating:", size=16),
            white_rating_slider,
            ft.Text("Black Rating:", size=16),
            black_rating_slider,
            ft.Text("Opening Ply:", size=16),
            opening_ply_dropdown,
            ft.Text("Victory Status:", size=16),
            victory_status_choices,
            ft.Text("Opening ECO:", size=16),
            opening_eco_choices,
            ft.Text("Increment Code:", size=16),
            increment_code_choices,
            boton_knn,
            contenedor_victoria_knn
        ]

    scrollable_content = ft.Column(
        items,
        expand=True,
        spacing=10,
        scroll="auto",
        
    )

    contenedor_predicciones=ft.Container(
                    content=scrollable_content,
                    alignment=ft.Alignment.TOP_CENTER,
                    bgcolor=ft.Colors.GREY,
                    expand=1,
                    

    )

    columna_interaccion=ft.Column([contenedor_metricas, contenedor_predicciones])



    contenedor_tablero = ft.Container(
                    content=columna_tablero,
                    margin=10,
                    alignment=ft.Alignment.TOP_CENTER,
                    bgcolor=ft.Colors.WHITE10,
                    width=page.width * width_percentage_tablero,
                    height=page.height * height_percentage_tablero,
                    border_radius=10,
                )
    
    contenedor_variables = ft.Container(
                    content=columna_interaccion,
                    margin=10,
                    padding=10,
                    alignment=ft.Alignment.TOP_CENTER,
                    bgcolor=ft.Colors.WHITE10,
                    width=page.width * width_percentage_variables,
                    height=page.height * height_percentage_variables,
                    border_radius=10,
                )
    
    contenido_linea=ft.Row([contenedor_tablero, contenedor_variables])

    
    
    # items=[contenido_linea]
    

    # columna_scroll = ft.Column(
    #     items,
    #     scroll="auto",  # Habilitar barra de desplazamiento vertical
    #     expand=True,    # Expandir para ocupar espacio disponible
    # )
##############################################################################################################
    
    ruta_metricas_tree = "tree_metrics_report.txt"
    try:
        with open(ruta_metricas_tree, "r") as file:
            metricas_tree = file.read()
    except FileNotFoundError:
        metricas_tree = "Metrics file not found. Please run the model first."

    boton_matriz_confusion_tree = ft.IconButton(
        icon=ft.Icons.IMAGE,  # Icon for the button
        icon_size=30,
        tooltip="Matriz de confusión",
        on_click=lambda _: change_image(page, contenedor_imagen, "img2/matriz_confusion_tree.png"),  # New image path
    )

    
    user_data_tree = {}

    
    def update_data_tree(e):
       
        user_data_tree['white_rating'] = white_rating_slider_tree.value
        user_data_tree['black_rating'] = black_rating_slider_tree.value
        user_data_tree['opening_ply'] = opening_ply_dropdown_tree.value
        user_data_tree['victory_status'] = victory_status_choices_tree.value
        user_data_tree['opening_eco'] = opening_eco_choices_tree.value
        user_data_tree['increment_code'] = increment_code_choices_tree.value
        
        print("Tree:" )
        print(user_data_tree)

       
        page.update()

    # Create sliders for the ratings
    white_rating_slider_tree = ft.Slider(
        min=784,
        max=2622,
        divisions=2200,
        label="{value}",
        value=1500,
        on_change=update_data_tree,  # Call the update_data function on change
    )

    black_rating_slider_tree = ft.Slider(
        min=784,
        max=2622,
        divisions=2200,
        label="{value}",
        value=1500,
        on_change=update_data_tree,  # Call the update_data function on change
    )

    # Create dropdown for opening play
    opening_ply_dropdown_tree = ft.Dropdown(
        options=[ft.dropdown.Option(str(i)) for i in range(1, 14)],
        value="1",
        on_select=update_data_tree,  # Call the update_data function on change
    )

    # Create RadioGroup for victory status (single choice)
    victory_status_choices_tree = ft.RadioGroup(
        content=ft.Column([
            ft.Radio("Victory Status Mate", value=1),
            ft.Radio("Victory Status Resign", value=2),
            
        ]),
        value="1",  # Default value
        on_change=update_data_tree,  # Call the update_data function on change
    )

    # Create RadioGroup for opening eco (single choice)
    opening_eco_choices_tree = ft.RadioGroup(
        content=ft.Column([
            ft.Radio("Opening ECO B00", value=1),
            ft.Radio("Opening ECO B01", value=2),
            ft.Radio("Opening ECO C00", value=3),
            ft.Radio("Opening ECO C41", value=4),
            ft.Radio("Opening ECO D00", value=5),
            ft.Radio("Other", value=6),
        ]),
        value=1,  # Default value
        on_change=update_data_tree,  # Call the update_data function on change
    )

    # Create RadioGroup for increment code (single choice)
    increment_code_choices_tree = ft.RadioGroup(
        content=ft.Column([
            ft.Radio("Increment Code 10+0", value=1),
            ft.Radio("Increment Code 15+0", value=2),
            ft.Radio("Other", value=3),
        ]),
        value=1,  # Default value
        on_change=update_data_tree,  # Call the update_data function on change
    )


    contenedor_imagen_tree=ft.Container(
                    content=ft.Image(
                        src=f"img2/matriz_confusion_tree.png",
                        expand=True,
                        fit=ft.BoxFit.CONTAIN,
                    ),
                    margin=10,
                    padding=10,
                    alignment=ft.Alignment.CENTER,
                    bgcolor=ft.Colors.GREY,
                    expand=9,
                    border_radius=10,
    )


    eleccion_opciones_tree=ft.Container(
                    content=ft.Row([boton_matriz_confusion_tree],
                                   alignment=ft.Alignment.CENTER
                                   ),
                    alignment=ft.Alignment.TOP_CENTER,
                    bgcolor=ft.Colors.GREY,
                    expand=1,

    )
    
    columna_tablero_tree=ft.Column([contenedor_imagen_tree, eleccion_opciones_tree])

    contenedor_metricas_tree=ft.Container(
                    content=ft.Text(
                        value=metricas_tree,
                        size=14,  # Font size for better readability
                        selectable=True,  # Allow the user to select and copy text
                    ),
                    alignment=ft.Alignment.TOP_CENTER,
                    bgcolor=ft.Colors.GREY,
                    expand=1,
                    

    )
    
    
    contenedor_victoria_tree=ft.Container(
        content=ft.Text(value="Sin predecir"),
        padding=20,
        bgcolor="blue",
        border_radius=15,
        border=ft.Border.all(3, "white"),
        alignment=ft.Alignment.CENTER,
        )
    
    
    def on_button_click_tree(e):
        lista_tree = f.random_values_interactive( 
        data_2
        ,white_rating_slider_tree.value
        ,black_rating_slider_tree.value
        ,opening_ply_dropdown_tree.value
        ,victory_status_choices_tree.value
        ,opening_eco_choices_tree.value
        ,increment_code_choices_tree.value
        )
        
        prediccion = model_tree.predict(lista_tree)
        if prediccion==1:
            contenedor_victoria_tree.content=ft.Text(value="Ganan las blancas", color=ft.Colors.BLACK)
            contenedor_victoria_tree.bgcolor=ft.Colors.WHITE
        else:
            
            contenedor_victoria_tree.content=ft.Text(value="Gana negro", color=ft.Colors.WHITE)
            contenedor_victoria_tree.bgcolor=ft.Colors.BLACK
        page.update()

    # Create a button and assign the click event
    boton_tree = ft.Button("Prediccion", on_click=on_button_click_tree)


    


    items_tree=[
            ft.Text("Chess Game Input Form", size=20, weight=ft.FontWeight.BOLD),
            ft.Text("White Rating:", size=16),
            white_rating_slider_tree,
            ft.Text("Black Rating:", size=16),
            black_rating_slider_tree,
            ft.Text("Opening Ply:", size=16),
            opening_ply_dropdown_tree,
            ft.Text("Victory Status:", size=16),
            victory_status_choices_tree,
            ft.Text("Opening ECO:", size=16),
            opening_eco_choices_tree,
            ft.Text("Increment Code:", size=16),
            increment_code_choices_tree,
            ft.Text("Boton:", size=16),
            boton_tree,
            contenedor_victoria_tree
        ]

    scrollable_content_tree = ft.Column(
        items_tree,
        expand=True,
        spacing=10,
        scroll="auto",
        
    )

    contenedor_predicciones_tree=ft.Container(
                    content=scrollable_content_tree,
                    alignment=ft.Alignment.TOP_CENTER,
                    bgcolor=ft.Colors.GREY,
                    expand=1,
                    
    )

    columna_interaccion_tree=ft.Column([contenedor_metricas_tree, contenedor_predicciones_tree])



    contenedor_tablero_tree = ft.Container(
                    content=columna_tablero_tree,
                    margin=10,
                    alignment=ft.Alignment.TOP_CENTER,
                    bgcolor=ft.Colors.WHITE10,
                    width=page.width * width_percentage_tablero,
                    height=page.height * height_percentage_tablero,
                    border_radius=10,
                )
    
    contenedor_variables_tree = ft.Container(
                    content=columna_interaccion_tree,
                    margin=10,
                    padding=10,
                    alignment=ft.Alignment.TOP_CENTER,
                    bgcolor=ft.Colors.WHITE10,
                    width=page.width * width_percentage_variables,
                    height=page.height * height_percentage_variables,
                    border_radius=10,
                )
    
    contenido_linea_tree=ft.Row([contenedor_tablero_tree, contenedor_variables_tree])

    page.on_resized = lambda e: set_size(e, page, contenedor_tablero, width_percentage_tablero, height_percentage_tablero, 
                                         contenedor_variables, width_percentage_variables, height_percentage_variables,
                                         contenedor_tablero_tree, contenedor_variables_tree)
    

    content_container = ft.Container(content=contenido_linea, expand=True)

    def btn1_clicked(e):
        content_container.content = contenido_linea
        page.update()

    def btn2_clicked(e):
        content_container.content = contenido_linea_tree
        page.update()

    btn1 = ft.Button("Modelo K-NN", icon=ft.Icons.IMAGE_ROUNDED, on_click=btn1_clicked)
    btn2 = ft.Button("Árboles de decisión", icon=ft.Icons.BAR_CHART, on_click=btn2_clicked)
    
    tabs_row = ft.Row([btn1, btn2], alignment=ft.Alignment.CENTER)

    page.add(ft.Column([tabs_row, content_container], expand=True))
    

ft.run(main)

