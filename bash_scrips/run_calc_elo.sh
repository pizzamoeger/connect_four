Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=:1
nohup ./bash_scrips/calc_elo.sh &
