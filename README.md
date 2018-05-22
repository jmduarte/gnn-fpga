# gnn-fpga

To log in to rulinux with port forwarding:
```
ssh -L localhost:9999:localhost:9999 jduarte1@rulinux04.DHCP.fnal.gov
```

To install:
```
source install_miniconda.ish
source install.sh
```

To setup:
```
source setup.sh
```

To run jupyter notebook over the same port:
```
jupyter notebook --ip 127.0.0.1 --port 9999 --no-browser 
```


