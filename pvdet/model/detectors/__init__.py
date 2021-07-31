from pvdet.model.detectors.part2net import Part2net


def build_netword(num_class,dataset,logger=None):

    model = Part2net(num_class=num_class,dataset=dataset)

    return model

