# Makefile 

# Instalar dependencias
install:
	pip install --upgrade pip setuptools
	pip install -r requirements.txt

# Formatear código
format:
	black *.py

# Entrenar modelo
train:
	python train.py

# Evaluar modelo y generar reporte
eval:
	python eval.py
	cat Results/metrics.txt >> report.md
	echo "" >> report.md
	echo "## Confusion Matrix Plot" >> report.md
	echo "![Confusion Matrix](./Results/model_results.png)" >> report.md
	cml comment create report.md

# Actualizar branch
update-branch:
	git config --global user.name '$(USER_NAME)'
	git config --global user.email '$(USER_EMAIL)'
	git add Results/
	git commit -am "Update with new results"
	git push --force origin HEAD:update

# Subir modelo a HuggingFace
hf-login:
	pip install -U "huggingface_hub[cli]"
	git pull origin main
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

# Subir archivos del modelo a HuggingFace
push-hub:
	huggingface-cli upload LuisaTirado/Drug-Classification ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload LuisaTirado/Drug-Classification ./Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload LuisaTirado/Drug-Classification ./Results/Metrics --repo-type=space --commit-message="Sync Metrics"

# Desplegar en HuggingFace
deploy:
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF_TOKEN)
	huggingface-cli upload LuisaTirado/Drug-Classification ./Model --repo-type=space --commit-message="Sync Model"