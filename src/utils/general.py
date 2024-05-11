

def free_subnet(subnet):
    for p in subnet.parameters():
        p.requires_grad = False