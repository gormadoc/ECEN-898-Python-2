Rem %windir%\System32\cmd.exe "/K" F:\Programs\Anaconda3_2020.11\Scripts\activate.bat F:\Programs\Anaconda3_2020.11

start /B conda run -n base main.py -v true -i coins -m 50
start /B conda run -n base main.py -v true -i coins -m 50 -c 8
start /B conda run -n base main.py -v true -i coins_g -m 50 -c 8
start /B conda run -n base main.py -v true -i coins_g -m 50
start /B conda run -n base main.py -v true -i coins_g -m 50 -n 5
start /B conda run -n base main.py -v true -i coins_g -m 50 -n 10
start /B conda run -n base main.py -v true -i coins_g -m 50 -n 15