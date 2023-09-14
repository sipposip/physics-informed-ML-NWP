
# for the highres sphereconv runs, the maximum tetralith timelimit of 7 days was too short.
# therefore, it was "boosted" to 14 days

for jid in 11550044 11550055 11550052 11550053 11550054 11550046 11550047 11550045; do
nsc-boost-timelimit -t 14-00:00:00 --accept-the-risks $jid
done