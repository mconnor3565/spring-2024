mkdir -p runs

cd runs
for i in {1..10}
do 

mkdir -p run${i}

cd run${i}

mkdir -p plots
mkdir -p data

cp ../../data.py . 
cp ../../reader.py .

python3 data.py 
python3 reader.py 

cd ../
done
cd ../  
