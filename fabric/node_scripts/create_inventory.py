ips = ["abc", "def", "ghi"]

inventory = (
    f"[kube_control_plane]\n"
    f"node1 ansible_host={ips[0]} ip={ips[0]} etcd_member_name=etcd1\n"
    f"\n"
    f"[etcd:children]\n"
    f"kube_control_plane\n"
    f"\n"
    f"[kube_node]\n"
    f"node2 ansible_host=10.145.5.3 ip=10.145.5.3\n"
    f"node3 ansible_host=10.145.5.4 ip=10.145.5.4\n"
)

for i, ip in enumerate(ips):
    inventory + f"node{i + 1} ansoble_host={ip} ip={ip}\n"
