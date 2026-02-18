import os 

# Estructura del proyecto

PROJECT_STRUCTURE = {
    "src": {
        "__init__.py": "",
        "components": {
            "__init__.py": "",
            "data_ingestion.py": "",
            "data_validation.py": "",
            "data_transformation.py": "",
            "model_trainer.py": "",
            "model_evaluation.py": "",
        },
        "exception": {
            "__init__.py": "",
            "exception.py": "",
        },
        "logging": {
            "__init__.py": "",
            "logger.py": "",
        },
        "pipeline": {
            "__init__.py": "",
            "main.py": "",
        },
  
    },
    # Archivos externos vacíos
    ".gitignore": "",
    "requirements.txt": "",
    "setup.py": "",

 
}

# Función recursiva para crear estructura

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)

        # Si es un diccionario → carpeta
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            # Si es string → archivo vacío
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

if __name__ == "__main__":
    print("Creando estructura del proyecto...")
    create_structure(".", PROJECT_STRUCTURE)
    print("Estructura generada correctamente.")
