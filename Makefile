.PHONY: randmst clean

randmst: randmst.py
	echo '#!/usr/bin/env python3' > randmst
	echo 'python3 randmst.py "$$@"' >> randmst
	chmod +x randmst

clean:
	rm -f randmst
