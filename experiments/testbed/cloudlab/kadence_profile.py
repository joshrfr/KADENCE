"""CloudLab profile for the KADENCE multi-node experiment (minimal/robust).

A head node + N workers on one LAN. No pinned disk image (use the cluster
default, which always exists for the chosen hardware type) and no boot-time
package installs (dependencies are installed by deploy/bootstrap.sh after the
nodes are ready, so a transient apt failure never fails provisioning). CloudLab
auto-assigns LAN addresses and populates /etc/hosts with the node names
(head, w000, ...), which the deploy scripts use.

Parameters: workers (count), wtype (worker hardware type), htype (head type).
Binding a hardware type that exists at only one cluster (e.g. xl170 -> Utah,
pc3000/d710 -> Emulab) routes the experiment to that cluster.
"""
import geni.portal as portal
import geni.rspec.pg as pg

pc = portal.Context()
pc.defineParameter("workers", "Number of worker nodes",
                   portal.ParameterType.INTEGER, 8)
pc.defineParameter("wtype", "Worker hardware type (blank = any)",
                   portal.ParameterType.STRING, "")
pc.defineParameter("htype", "Head hardware type (blank = any)",
                   portal.ParameterType.STRING, "")
params = pc.bindParameters()

request = pc.makeRequestRSpec()
lan = request.LAN("lan")


def mk(name, htype):
    n = request.RawPC(name)
    if htype:
        n.hardware_type = htype
    lan.addInterface(n.addInterface("if0"))
    return n


mk("head", params.htype)
for i in range(params.workers):
    mk("w%03d" % i, params.wtype)

pc.printRequestRSpec(request)
