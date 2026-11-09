from pathlib import Path

path = Path("job_mpc_product.pbs")
s = path.read_text()

old = '''echo "Status final: $STATUS"
exit "$STATUS"'''

new = '''# Conferir as 50 metricas finais
COUNT=$(find "$ROOT/results" -name metrics.json 2>/dev/null | wc -l)

echo "Treinamentos concluidos: $COUNT/50"

if [ "$COUNT" -ne 50 ]; then
    STATUS=1
fi

echo "Status final: $STATUS"
exit "$STATUS"'''

assert old in s
path.write_text(s.replace(old, new))
