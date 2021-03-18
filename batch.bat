%windir%\System32\cmd.exe "/K" D:\Programs\Anaconda\Scripts\activate.bat D:\Programs\Anaconda
start /B conda run -n base main.py -v true -i test -c 4
start /B conda run -n base main.py -v true -i elk -c 4
start /B conda run -n base main.py -v true -i elk -c 8
start /B conda run -n base main.py -v true -i elk -c 4 -m 50
start /B conda run -n base main.py -v true -i elk -c 8 -m 50
start /B conda run -n base main.py -v true -i elk -c 4 -m 25
start /B conda run -n base main.py -v true -i elk -c 4 -m 10
start /B conda run -n base main.py -v true -i elk -c 4 -m 50 -s 9
start /B conda run -n base main.py -v true -i elk -c 4 -m 50 -s 3
start /B conda run -n base main.py -v true -i elk -c 4 -m 50 -s 1