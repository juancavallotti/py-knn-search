# KNN Search for Python

### Release Notes
* Version 0.1.0: I added the ability to implement an alternative index storage backend through the ```IndexBackend``` interface. With this feature we can read and store the information at different locations and not just a static pickle file.


This is my artisanal attempt at KNN Search. The intended use of this library is for generic symmetric contextual search and works with various word embeddings through the ```Embedder``` interface.  I use this library throughout my projects, and it’s distributed as-is, according to the terms of its license. I use it on production systems, but please, use it at your own risk.