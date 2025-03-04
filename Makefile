.PHONY: randmst clean

randmst: randmst.py
	echo '#!/bin/bash' > randmst
	echo 'python3 randmst.py "$$@"' >> randmst
	chmod +x randmst

clean:
	rm -f randmst
