randmst: randmst.py
	printf '#!/usr/bin/env python\n' >$@
	python ./$< >$@
	chmod +x $@