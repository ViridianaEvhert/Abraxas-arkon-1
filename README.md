# Abraxas-arkon-1
Test-Viridiana-Barragan
El algoritmo empleado para la predecir una de serie de tiempo fue Autoregression Model.

Para correr el algoritmo:

1. DESCARGANDO EL ARCHIVO SCRIPT .PY Y EL ARCHIVO .CSV. CORRIENDO EN TERMINAL

-Guardar py3_temp_vs_time.py en alguna carpeta junto con el dataset daily-min-temperatures.csv

-Abrir una terminal y pararse en la carpeta dónde está el script .py y el dataset .csv

-Correr desde terminal: pip install numpy scipy pandas matplotlib scikit-learn statsmodels 

-Correr desde terminal la siguiente línea: chmod 777 py3_temp_vs_time.py

-Correr desde terminal la siguiente línea: python3.5 py3_temp_vs_time.py

-Se creará una carpeta llamada Results, que contiene las gráficas para el análisis de la serie de tiempo y la predicción (El scrip está comentado con una breve explicación de los pasos)

-En terminal se imprimen resultados como: lag, Coefficients, predicted, expected, etc


2. DOCKERFILE (MANUAL)

-Crear el directorio imagen-python:
mkdir imagen-python

-Posicionarse en el directorio imagen-python:
cd imagen-python

-Guardar en el directorio imagen-python el script py3_temp_vs_time.py y el archivo daily-min-temperatures.csv

-Crear el Dockerfile y editarlo:

vi Dockerfile 

FROM python:3.6.8
ADD py3_temp_vs_time.py /
ADD daily-min-temperatures.csv /
RUN pip install numpy scipy pandas matplotlib scikit-learn statsmodels
CMD ["python", "./py3_temp_vs_time.py"]

-Construir la imagen :

sudo docker build -t python_test_viridiana .

-Correr para que ejecute el scrip de python

sudo docker run -it python_test_viridiana 



