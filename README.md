# Usage

## Docker
```
docker build -t pbrdf:latest -f Dockerfile .
docker run --rm -itd --name=pbrdf -v ~/projects/pbrdf-rendering:/pbrdf-rendering pbrdf:latest
docker exec -it pbrdf bash
```

## Download pBRDF database
```
sh scripts/download_pbrdf_database.sh path/to/directory/ supp_info/obj_pbrdf.txt
```

## Rendering sphere
```
python3 rendering.py --pbrdf_dir path/to/pbrdf/database/dir --N_map_file supp_info/N_map_100.npy --mask_file supp_info/mask_100.png --L_file supp_info/L_4.txt --stokes_file supp_info/stokes_L_4.txt --out_dir path/to/output/dir --obj_file supp_info/obj_pbrdf.txt --save_type png --obj_range 0 25 --n_jobs -1
```
