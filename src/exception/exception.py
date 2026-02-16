# Importa el módulo sys, necesario para obtener información detallada del error
import sys


# Define una excepción personalizada que hereda de Exception
class SalesForecastException(Exception):

    # Constructor que recibe el mensaje de error y los detalles del sistema
    def __init__(self, error_message, error_details: sys):

        # Guarda el mensaje original del error
        self.error_message = error_message

        # Obtiene la información del traceback (pila de llamadas)
        _, _, exc_tb = error_details.exc_info()

        # Extrae el número de línea donde ocurrió el error
        self.lineno = exc_tb.tb_lineno

        # Extrae el nombre del archivo donde ocurrió el error
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    
    # Método que define cómo se imprime la excepción personalizada
    def __str__(self):
        # Devuelve un mensaje formateado con archivo, línea y descripción del error
        return (
            "Error occured in python script name [{0}] line number [{1}] error message [{2}]"
            .format(self.file_name, self.lineno, str(self.error_message))
        )
    
# Punto de entrada del script
if __name__ == '__main__':

    # Bloque try para capturar errores
    try:

        # Provoca un error de división por cero
        a = 1 / 0

        # Esta línea nunca se ejecutará debido al error anterior
        print("This will not be printed", a)

    # Captura cualquier excepción que ocurra
    except Exception as e:

        # Lanza la excepción personalizada con detalles del error
        raise SalesForecastException(e, sys)




