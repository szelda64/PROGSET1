#! /usr/bin/env python
all: randmst.py
	printf '#!/usr/bin/env python\n' >$@
	cp randmst.py randmst && chmod +x randmst && ./randmst