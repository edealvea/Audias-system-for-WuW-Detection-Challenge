# TFM

Repostitorio conteniendo el código utilizado para entrenar el modelo presentado para las 2024 Albayzin Evaluations, en concreto para el Wake-up Word detection challenge propuesto por Telefónica. El challenge consiste en crear un modelo de keyword spotting para detectar la frase "Okey Aura" en un segmento de audio y además determinar los tiempos en los que se ha dicho la frase.

**No se van a compartir los datos utilizados para dicho entrenamiento por estar sujetos a confidencialidad.**


## Archivos

- models.py es el archivo que contiene las estructuras de los modelos estudiadas para el challenge
- model_trainer.py archivo que cuenta con todo el código utilizado para entrenar, probar y guardar el modelo
- Evaluate_model.py contiene el código utilizado para evaluar el modelo
- gen_evaluation.py contiene el código para generar los resultados del modelo para un conjunto de datos dado.
- test_system.py contiene código para testear un modelo en un conjunto de evaluación.
- notebook.ipynb es un notebook para utilizar el código anterior
- run_gen_evaluation.sh archivo bash para ejecutar gen_evaluation.py
- run_tests.sh bach para ejecutar test_system.py
- AUDIAS_System_for_the_ALBAYZIN_2024_WuW_Detection_Challenge.pdf es el paper publicado con los resultados obtenidos en el challenge

  
