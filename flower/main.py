import subprocess


# Iterable con las rutas de los scripts
scripts_paths = ("server.py", "client.py", "client.py")

result=subprocess.run(["c:/Users/DMartinez/Documents/Repositories/FedEnhancedDataSynthesizer/.venv/Scripts/activate.bat", "python", "client.py"], shell=True, capture_output=True, text=True)
print(result)
subprocess.run(["python server.py"], shell=True, capture_output=True, text=True)
# Creamos cada proceso    
#procesos = [subprocess.Popen( f"c:/Users/DMartinez/Documents/Repositories/FedEnhancedDataSynthesizer/.venv/Scripts/activate.bat python {script}" ) for script in scripts_paths]

# Esperamos a que todos los subprocesos terminen.
# for proceso in procesos:
#     proceso.wait()

# Resto de código a ejecutar cuando terminen todos los subprocesos.