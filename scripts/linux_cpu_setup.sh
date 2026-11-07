#!/bin/sh
# One-time setup on the Linux host (ArmadaOS on the AYN Thor and the like): lets play.sh put the CPU at full speed while a game runs and back afterwards, without a
# password. Installs /etc/nxnative/nxnative-cpu (on / off) and a sudoers rule allowing only that helper for your user.
#   sudo sh scripts/linux_cpu_setup.sh            install
#   sudo sh scripts/linux_cpu_setup.sh --remove   remove both
set -e
[ "$(id -u)" = 0 ] || { echo "run it with sudo: sudo sh $0"; exit 1; }
[ ! -f /run/.containerenv ] || { echo "run it on the host, not inside the toolbox (type exit first)"; exit 1; }
helper=/etc/nxnative/nxnative-cpu rule=/etc/sudoers.d/nxnative-cpu
if [ "$1" = --remove ]; then
    [ -x "$helper" ] && "$helper" off || true
    rm -f "$helper" "$rule"; echo "removed $helper and $rule"; exit 0
fi
user="${SUDO_USER:-}"; [ -n "$user" ] && [ "$user" != root ] || { echo "run it with sudo from your own account"; exit 1; }
mkdir -p /usr/local/bin
cat > "$helper" <<'H'
#!/bin/sh
# nxnative: the CPU's speed setting for a game session. on: the performance governor (the setting before kept in /run/nxnative-cpu); off: that setting back
st=/run/nxnative-cpu
case "$1" in
on)  [ -f "$st" ] || cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor > "$st"
     for p in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > "$p/scaling_governor"; done ;;
off) g="$(cat "$st" 2>/dev/null || echo schedutil)"
     for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "$g" > "$p/scaling_governor"; done; rm -f "$st" ;;
*)   echo "usage: nxnative-cpu on|off"; exit 2 ;;
esac
H
chown root:root "$helper"; chmod 755 "$helper"
printf '%s ALL=(root) NOPASSWD: %s on, %s off\n' "$user" "$helper" "$helper" > "$rule.tmp"
chmod 440 "$rule.tmp"
if visudo -cf "$rule.tmp" >/dev/null; then mv "$rule.tmp" "$rule"; else rm -f "$rule.tmp"; echo "the sudoers rule did not check out: nothing installed"; rm -f "$helper"; exit 1; fi
echo "installed: $helper and $rule (for $user). Rebuild once so play.sh uses it; DM_CPU_FULL=0 in ~/.config/nxnative/env turns it off"
