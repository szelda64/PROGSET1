PY=python3

randmst: randmst.py rehualPrim.py genGraphRehaul.py
	python3 randmst.py rehualPrim.py genGraphRehaul.py
	chmod +x randmst.py
