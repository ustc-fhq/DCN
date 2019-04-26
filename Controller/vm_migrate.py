import subprocess
dict={'0':'172.16.0.218','1':'172.16.0.219','2':'172.16.0.221'}
vm_name=["instance-00000053","instance-00000055","instance-00000057","instance-0000006b","instance-0000006c","instance-0000006f","instance-00000067","instance-00000068","instance-00000069"]
#vm=["master1","node1","node2","master2","node3","node4","master3","node5","node6"]
vm_uuid=["c8a445af-120b-4c6f-b582-d76f066b12fb","8091bfcd-0191-42ab-9dc9-8b1981405d93","e8015020-b051-4861-a6b8-9f2aad1468fe","6817f533-ce6b-42f4-823e-046000a3646e","8baa8185-36af-483f-9a1b-b7e9808b2f01","3ec8e7f8-3c3c-4cb5-becc-233ea05e825b","2a9da954-3389-4b83-9b47-5d540cdf5402","5bc17cd5-32cf-4b37-9c48-b2d31e6c4af4","b2a09bfd-f8b7-45be-935f-2d79a09d4eb8"]
vm_ip=["10.10.10.11","10.10.10.24","10.10.10.15","10.10.10.20","10.10.10.9","10.10.10.26","10.10.10.7","10.10.10.22","10.10.10.5",]
def vm_migrate(vm_num,src_host,dst_host):

    out=subprocess.getoutput('virsh  -c  qemu+tcp://'+dict[src_host]+'/system migrate --live --undefinesource --persistent --tunnelled --p2p '+vm_name[vm_num]+' qemu+tcp://'+dict[dst_host]+'/system tcp://'+dict[dst_host])
    print(out)

#def refresh(vm_num,dst_host):
# out=subprocess.getoutput('mysql -uroot -e '+'\"'+'update nova.instances set host='+'\''+dst_host+'\''+',lauched_on='+'\''+dst_host+'\''+',node='+'\''+dst_host+'\''+' where uuid='+'\''+vm_uuid[vm_num]+'\''+';'+'\"')
#out=subprocess.getoutput('mysql -uroot -e '+'\"'+'update neutron.ml2_port_bindings set host='+'\''+dst_host+'\''+' where port_id=(select port_id from neutron.ipallocations where ip_address='+'\''+vm_ip[vm_num]+'\''+');'+'\"')
#out=subprocess.getoutput('mysql -uroot -e '+'\"'+'update neutron.ports set status='+'\''+'ACTIVE'+'\''+' where id=(select port_id from neutron.ipallocations where ip_address='+'\''+vm_ip[vm_num]+'\''+');'+'\"')


