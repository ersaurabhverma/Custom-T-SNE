# customTSNE

customTSNE is a basic implementation of TSNE dimension reduction technique.



## Usage

```python
from customTSNE import cusomTSNE # Importing the customTSNE class

tsne=customTSNE() # Initializing the instance
li=tsne.run(data[['x','y']])
print(li[-1]) # Getting the reduced dimesion of last iteration
 
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
