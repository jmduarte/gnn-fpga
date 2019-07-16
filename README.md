# gnn-fpga

To log in to `cmslpc` with port forwarding:
```bash
ssh -L localhost:9999:localhost:9999 YOUR_USERNAME@cmslpc-sl6.fnal.gov
```

To install:
```
cd ~/nobackup # ~/nobackup area has more space
git clone https://github.com/jmduarte/gnn-fpga
cd gnn-fpga
source install_miniconda.sh
source install.sh
```

To setup (each time you log in):
```
source setup.sh
```

To run jupyter notebook over the same port (and keep it in background):
```
jupyter notebook --ip 127.0.0.1 --port 9999 --no-browser &
```

Then on your laptop (or local computer) you can open the provided link in your browser, which looks like (for example):
```
http://127.0.0.1:9999/?token=4c6353079bb24900167910cb653bbcdbf16d1d8fff16d604
```

To run the muon graph creation (for example) do:
```
python prepareMuonGraphs.py --input-muon-dir /eos/uscms/store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleMuon_Endcap_2GeV/ParticleGuns/CRAB3/190416_194707/0000/ --input-pu-dir /eos/uscms/store/group/l1upgrades/L1MuonTrigger/P2_10_4_0/ntuple_SingleNeutrino_PU200/SingleNeutrino/CRAB3/190416_160207/0000/
```
