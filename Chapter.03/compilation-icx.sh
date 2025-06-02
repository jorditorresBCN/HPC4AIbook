icx -O0 test.c -o test
echo "icx -O0"
./test
icx -O1 test.c -o test
echo "icx -O1"
./test
icx -O2 test.c -o test
echo "icx -O2"
./test
icx -O3 test.c -o test
echo "icx -O3"
./test
icx -O3 -xhost test.c -o test
echo "icx -O3 -xhost "
./test
