#!/bin/bash

number=(500 500 500)

./matmult_c.gcc nat $number[0] $number[1] $number[2]


./matmult_c.gcc mnk $number[0] $number[1] $number[2]

./matmult_c.gcc mkn $number[0] $number[1] $number[2]

./matmult_c.gcc nmk $number[0] $number[1] $number[2]

./matmult_c.gcc nkm $number[0] $number[1] $number[2]

./matmult_c.gcc kmn $number[0] $number[1] $number[2]

./matmult_c.gcc knm $number[0] $number[1] $number[2]

