#!/bin/bash

module load studio

arr=(50 850 1600)

echo "\nnow ready for performance test\n"

echo "\nstart matmult_mnk\n"
for i in "${arr[@]}"
do
    matmult_c.gcc mnk $i $i $i
    collect -o test.mnk.1.er -p on -S on -h dch -h dcm -h l2h -h l2m matmult_c.gcc mnk $i $i $i
done
echo "\nend matmult_mnk\n"

echo "\nstart matmult_mkn\n"
for i in "${arr[@]}"
do
    matmult_c.gcc mkn $i $i $i
    collect -o test.mkn.1.er -p on -S on -h dch -h dcm -h l2h -h l2m matmult_c.gcc mkn $i $i $i
done
echo "\nend matmult_mkn\n"

echo "nstart matmult_nmk\n"
for i in "${arr[@]}"
do
    matmult_c.gcc nmk $i $i $i
    collect -o test.nmk.1.er -p on -S on -h dch -h dcm -h l2h -h l2m matmult_c.gcc nmk $i $i $i
done
echo "\nend matmult_nmk\n"

echo "\nstart matmult_nkm\n"
for i in "${arr[@]}"
do
    matmult_c.gcc nkm $i $i $i
    collect -o test.nkm.1.er -p on -S on -h dch -h dcm -h l2h -h l2m matmult_c.gcc nkm $i $i $i
done
echo "\nend matmult_nkm\n"

echo "\nstart matmult_kmn\n"
for i in "${arr[@]}"
do
    matmult_c.gcc kmn $i $i $i
    collect -o test.kmn.1.er -p on -S on -h dch -h dcm -h l2h -h l2m matmult_c.gcc kmn $i $i $i
done
echo "\nend matmult_kmn\n"

echo "nstart matmult_knm\n"
for i in "${arr[@]}"
do
    matmult_c.gcc knm $i $i $i
    collect -o test.knm.1.er -p on -S on -h dch -h dcm -h l2h -h l2m matmult_c.gcc knm $i $i $i
done
echo "\nend matmult_knm\n"

er_print -func test.mnk.1.er > test.mnk.1.txt
er_print -func test.mnk.2.er > test.mnk.2.txt
er_print -func test.mnk.3.er > test.mnk.3.txt

er_print -func test.mkn.1.er > test.mkn.1.txt
er_print -func test.mkn.2.er > test.mkn.2.txt
er_print -func test.mkn.3.er > test.mkn.3.txt

er_print -func test.nmk.1.er > test.nmk.1.txt
er_print -func test.nmk.2.er > test.nmk.2.txt
er_print -func test.nmk.3.er > test.nmk.3.txt

er_print -func test.nkm.1.er > test.nkm.1.txt
er_print -func test.nkm.2.er > test.nkm.2.txt
er_print -func test.nkm.3.er > test.nkm.3.txt

er_print -func test.kmn.1.er > test.kmn.1.txt
er_print -func test.kmn.2.er > test.kmn.2.txt
er_print -func test.kmn.3.er > test.kmn.3.txt

er_print -func test.knm.1.er > test.knm.1.txt
er_print -func test.knm.2.er > test.knm.2.txt
er_print -func test.knm.3.er > test.knm.3.txt


echo "\n end of performance test\n"

