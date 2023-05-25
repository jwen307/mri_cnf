#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_multiscale_nets.py
    - Functions to build the conditional normalizing flow architecture
"""

#%%
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import sys
sys.path.append('../..')

from models.networks.inn_modules import misc
from models.networks.conditional_nets import resnet_conditional
from models.networks import buildingblocks as nets

# Build the single-coil CNF architecture described in https://github.com/jleuschn/cinn_for_imaging
def buildnetsc(img_size=[1,320,320], **kwargs):
    """
    Args:
        img_size=[1,320,320]
    Kwargs:
        num_downsample = 6
        cond_conv_chs = [4,16,16,16,32,32,32]
        downsample='squeeze'
        num_blocks = 5
        use_fc_block=False,
        num_fc_blocks=2,
        cond_fc_size=64
    """

    # Collect the conditional nodes and split nodes
    conditions = []
    split_nodes = []

    #Create all the conditional nodes
    for i in range(kwargs['num_downsample']):
        conditions.append(Ff.ConditionNode(kwargs['cond_conv_chs'][i],
                                           img_size[1]/(2**(i+1)),
                                           img_size[2]/(2**(i+1)),
                                           name='cond_{}'.format(i)
                                           ))

    if kwargs['use_fc_block']:
        conditions.append(Ff.ConditionNode(kwargs['cond_fc_size'],
                                           name='cond_{}'.format(kwargs['num_downsample'])))


    #Build the flow
    nodes = [Ff.InputNode(img_size[0], img_size[1], img_size[2], name='Input')]

    #First downsample (cin, h, w) -> (4*cin, h/2, w/2)
    nets._add_downsample(nodes, kwargs['downsample'])

    for k in range(kwargs['num_downsample']-1):
        #1) Add a coupling block
        nets._add_conditioned_section(nodes,
                                      downsampling_level=k,
                                      num_blocks=kwargs['num_blocks'],
                                      cond=conditions[k],
                                      coupling_type='affine',
                                      act_norm=True,
                                      permutation_type='1x1')

        #2) Downsample
        nets._add_downsample(nodes, kwargs['downsample'])

        #3) Split
        nodes.append(Ff.Node(nodes[-1],
                             misc.Split,
                             {},
                             name='split_{}'.format(k)
                             ))
        split_nodes.append(Ff.Node(nodes[-1].out1,
                                   Fm.Flatten,
                                   {},
                                   name='flatten_split.{}'.format(k)))


    #Flatten
    nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten,{}, name='conv_flatten'))

    #Add the fully connected block if needed
    if kwargs['use_fc_block']:

        nodes.append(Ff.Node(nodes[-1], misc.Split,
                             {'section_sizes': [128], 'dim':0, 'n_sections':2},
                             name='split_fc'))

        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten,
                                   {},
                                   name='flatten_split.fc'))

        nets._add_fc_section(nodes, cond=conditions[-1],
                             num_blocks=kwargs['num_fc_blocks'],
                             coupling_type='affine')

        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten,{}, name='final_flatten'))


    #Concatenate all of the split nodes
    nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                         Fm.Concat1d, {'dim':0}, name='concat'))

    #Define the output node
    nodes.append(Ff.OutputNode(nodes[-1], name = 'out'))

    #Create the flow network
    flow = Ff.GraphINN(nodes + conditions + split_nodes, verbose=False)


    #Create the conditional network
    bij_cnet = resnet_conditional.ResNetCondNet(img_size,
                                                downsample_levels=kwargs['num_downsample'],
                                                cond_conv_channels=kwargs['cond_conv_chs'],
                                                use_fc_block=kwargs['use_fc_block'],
                                                cond_fc_size=kwargs['cond_fc_size'],
                                                )


    #Return the normalizing flow and the conditioning network
    return flow, bij_cnet



# Build our conditional normalizing flow for the multicoil case
def buildnetmc(img_size=[16, 320, 320], **kwargs):
    """
        Args:
            img_size=[16,320,320]
        Kwargs:
            num_downsample = 3
            cond_conv_chs = [64,64,128]
            downsample='squeeze'
            num_blocks = 20
            use_fc_block=False,
            num_fc_blocks=2,
            cond_fc_size=64
    """
    conditions = []
    split_nodes = []

    # Create all the conditional nodes
    for i in range(kwargs['num_downsample']):
        conditions.append(Ff.ConditionNode(kwargs['cond_conv_chs'][i],
                                           img_size[1] / (2 ** (i + 1)),
                                           img_size[2] / (2 ** (i + 1)),
                                           name='cond_{}'.format(i)
                                           ))


    if kwargs['use_fc_block']:
        conditions.append(Ff.ConditionNode(kwargs['cond_fc_size'],
                                           name='cond_{}'.format(kwargs['num_downsample'])))

    # Build the flow
    nodes = [Ff.InputNode(img_size[0], img_size[1], img_size[2], name='Input')]

    for k in range(kwargs['num_downsample'] - 1):
        # 1) Downsample
        nets._add_downsample(nodes, kwargs['downsample'])

        # 2) Add a transition level to prevent checkboarding effects (SRFlow)
        nets.add_level_transition(nodes, downsampling_level=k)

        # 2) Add a coupling block
        nets._add_conditioned_section_new(nodes,
                                        downsampling_level=k,
                                        num_blocks=kwargs['num_blocks'],
                                        cond=conditions[k],
                                        coupling_type='affine',
                                        act_norm=True,
                                        permutation_type='1x1')

        # 3) Split
        nodes.append(Ff.Node(nodes[-1],
                             misc.Split,
                             {},
                             name='split_{}'.format(k)
                             ))
        split_nodes.append(Ff.Node(nodes[-1].out1,
                                   Fm.Flatten,
                                   {},
                                   name='flatten_split_{}'.format(k)))

    # Flatten
    nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {}, name='conv_flatten'))

    # Add the fully connected block if needed
    if kwargs['use_fc_block']:
        nodes.append(Ff.Node(nodes[-1], misc.Split,
                             {'section_sizes': [128], 'dim': 0, 'n_sections': 2},
                             name='split_fc'))

        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten,
                                   {},
                                   name='flatten_split_fc'))

        nets._add_fc_section(nodes, cond=conditions[-1],
                             num_blocks=kwargs['num_fc_blocks'],
                             coupling_type='affine')

        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {}, name='final_flatten'))

    # Concatenate all of the split nodes
    nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                         Fm.Concat1d, {'dim': 0}, name='concat'))

    nodes.append(Ff.OutputNode(nodes[-1], name='out'))

    # Create the flow network
    flow = Ff.GraphINN(nodes + conditions + split_nodes, verbose=False)

    # Create the new conditional network
    bij_cnet = resnet_conditional.ResNetCondNetProg(img_size,
                                                     downsample_levels=kwargs['num_downsample'],
                                                     cond_conv_channels=kwargs['cond_conv_chs'],
                                                     use_fc_block=kwargs['use_fc_block'],
                                                     cond_fc_size=kwargs['cond_fc_size'],
                                                     )

    return flow, bij_cnet

