# pylint: disable=invalid-name, consider-using-enumerate, missing-function-docstring

from numba import njit
from numba.types import int32
import numpy as np

# from numba import objmode
# import time
@njit([(int32[:], int32[:])], cache=True)
def get_in_degree(end_neighbors, neighbors):
    """Compute the in degree"""
    #n_nodes = len(end_neighbors)-1
    in_degree = np.zeros(len(end_neighbors)-1, dtype=np.int32)

    for i in range(len(neighbors)):
        in_degree[neighbors[i]]+=1

    return in_degree


@njit(cache=True)
def get_degree_partition(in_degree, max_degree):
    n_nodes = len(in_degree)
    num_per_degree = np.zeros(max_degree+2, dtype=np.int32)
    for i in range(len(in_degree)):
        num_per_degree[in_degree[i]]+=1



    start_nodes_by_class = np.empty(n_nodes, dtype=np.int32)
    end_nodes_by_class = np.empty(n_nodes, dtype=np.int32)
    prev_end = 0
    num_classes = 0
    for i in range(0, len(num_per_degree)):
        num_nodes_this_class = num_per_degree[i]
        if num_nodes_this_class == 0:
            continue
        num_per_degree[i] = prev_end

        start_nodes_by_class[num_classes] = prev_end
        end_nodes_by_class[num_classes] = prev_end + num_nodes_this_class
        prev_end += num_nodes_this_class
        num_classes+=1

    position_of_node = np.empty(n_nodes, dtype=np.int32)
    nodes_by_class = np.empty(n_nodes, dtype=np.int32)
    for i in range(n_nodes):
        nodes_by_class[num_per_degree[in_degree[i]]]=i
        position_of_node[i] = num_per_degree[in_degree[i]]
        num_per_degree[in_degree[i]]+=1


    classes=np.empty(n_nodes, dtype=np.int32)
    for i in range(num_classes):
        for j in range(start_nodes_by_class[i], end_nodes_by_class[i]):
            classes[nodes_by_class[j]]=i


    class_costs = np.zeros(num_classes, dtype=np.int32)
    for i in range(num_classes):
        num_nodes = end_nodes_by_class[i] - start_nodes_by_class[i]
        num_edges = in_degree[start_nodes_by_class[i]]
        class_costs[i] = num_edges * num_nodes

    queue = np.empty((n_nodes,2), dtype=np.int32)
    class_order = np.argsort(class_costs)[::-1]
    for i in range(num_classes):
        c = class_order[i]
        queue[i,0] = start_nodes_by_class[c]
        queue[i,1] = end_nodes_by_class[c]

    return num_classes, start_nodes_by_class, end_nodes_by_class, nodes_by_class, classes, position_of_node, queue



@njit([(int32[:], int32[:], int32[:])], cache=True)
def color_refinement_nlogn(end_neighbors, neighbors, initial_labels):
    """Compute the coarsest WL refinement"""
    # print("neighbprs")
    # print(end_neighbors)
    # print(neighbors)
    n_nodes = len(end_neighbors)-1
    max_degree = n_nodes
    starts_from_degree = np.all(initial_labels==initial_labels[0])
    if starts_from_degree:
        # with objmode(t1='double'):  # annotate return type
        #     t1 = time.process_time()
        in_degree = get_in_degree(end_neighbors, neighbors)
        num_classes, start_nodes_by_class, end_nodes_by_class, nodes_by_class, classes, position_of_node, queue = get_degree_partition(in_degree, in_degree.max())
        # with objmode(t2='double'):  # annotate return type
        #     t2 = time.process_time()
        # print(t2-t1)
        depth=1
    else:
        # print("initial", initial_labels)
        num_classes, start_nodes_by_class, end_nodes_by_class, nodes_by_class, classes, position_of_node, queue = get_degree_partition(initial_labels,  initial_labels.max())
        depth = 0
        # num_classes=1
        # start_nodes_by_class = np.empty(n_nodes, dtype=np.int32)
        # start_nodes_by_class[0] = 0
        # end_nodes_by_class = np.empty(n_nodes, dtype=np.int32)
        # end_nodes_by_class[0] = n_nodes
        # position_of_node = np.arange(n_nodes, dtype=np.int32)
        # nodes_by_class = np.arange(n_nodes, dtype=np.int32)
        # queue = np.empty(n_nodes, dtype=np.int32)
        # queue[0]=0
    #original = np.arange(len(neighbors))
    #where = np.arange(len(neighbors))
    out_classes = np.empty((n_nodes,3), dtype=np.int32)
    for i in range(num_classes):
        out_classes[i,0]=start_nodes_by_class[i]
        out_classes[i,1]=end_nodes_by_class[i]
        out_classes[i,2]=depth

    start_neighbors = end_neighbors[:-1].copy()
    end_neighbors = end_neighbors[1:]


    # per node characteristics
    #classes = np.zeros(n_nodes, dtype=np.int32)
    receive_counts = np.zeros(n_nodes, dtype=np.int32)
    node_is_active = np.zeros(n_nodes, dtype=np.bool_)
    #position_of_node = np.arange(n_nodes, dtype=np.int32)

    # nodes per class
    #start_nodes_by_class = np.empty(n_nodes, dtype=np.int32)
    #start_nodes_by_class[0] = 0 # first class contains all nodes
    #end_nodes_by_class = np.empty(n_nodes, dtype=np.int32)
    #end_nodes_by_class[0] = n_nodes # first class contains all nodes
    #nodes_by_class = np.arange(n_nodes)
    received_nodes_by_class = np.empty(n_nodes, dtype=np.int32)

    # queue
    #queue = np.empty(n_nodes, dtype=np.int32)
    #queue[0:num_classes]=np.arange(num_classes)
    queue_R = num_classes
    queue_L = 0
    class_in_queue = np.zeros(n_nodes, dtype=np.bool_)
    class_in_queue[:num_classes]=True
    classes_processed = 0





    #num_classes = 1
    active_nodes_in_class = np.zeros(n_nodes, dtype=np.int32) # counts the number of nodes that are affected by message passing of current class
    active_classes = np.empty(n_nodes, dtype=np.int32)# max_size: if all nodes have unique degree active_classes = n_nodes-1
    num_active_classes = 0

    # per group statistics
    group_ids = np.zeros(max_degree+1, dtype=np.int32)# max_size is upper bounded by max_degree+1 as no node can have count > max_degree
    num_nodes_per_group_scattered = np.zeros(max_degree+1, dtype=np.int32) # max_size: see above
    num_nodes_per_group = np.zeros(max_degree+1, dtype=np.int32) # max_size: see above
    #nodes_in_group = np.zeros(n_nodes, dtype=np.int32) # max_size=n_nodes because it is full in first iteration, afterwards could be max_i(num_nodes_with_degree=i * i)
    nodes_indices_by_group = np.zeros(max_degree+1, dtype=np.int32)
    class_name_for_group = np.zeros(max_degree+1, dtype=np.int32)

    # # performance metrics
    # num_messages = 0
    # num_groups_1 = 0
    # num_groups_2 = 0
    # num_groups_3 = 0
    # num_groups_x = 0
    # min_group_id_0 =0
    # min_group_id_1 =0
    # min_group_id_x =0
    # max_group_id_1 =0
    # max_group_id_2 =0
    # max_group_id_x =0
    # min_max_group_id_0_1 = 0
    # largest_group_id_0 = 0
    # special_case = 0
    # normal_case = 0

    # neighborhood_10 = 0
    # neighborhood_x = 0
    # all_unique = 0
    # num_active_1=0
    # num_active_2=0
    while queue_L < queue_R and num_classes < n_nodes:
        # print("queue", queue, queue_L, queue_R)
        # print(start_nodes_by_class)
        # print(end_nodes_by_class)
        # print(nodes_by_class)
        depth+=1
        last_queue_R = queue_R
        while queue_L < last_queue_R and num_classes < n_nodes:
            # print("queue", queue, queue_L, queue_R, last_queue_R)
            #print(queue_len)
            start_class = queue[queue_L,0]
            end_class = queue[queue_L,1]
            #assert send_class < num_classes
            #class_in_queue[send_class]=False
            queue_L+=1

            # Performance tracking
            classes_processed +=1
            # if classes_processed == 2:
            #     print("initial_messages", num_messages)

            #print("class", classes)
            #print("sending class", send_class)
            #print("start_nodes_by_class", start_nodes_by_class[send_class])
            #print("end_nodes_by_class", end_nodes_by_class[send_class])
            #print("nodes_by_class", nodes_by_class[start_nodes_by_class[send_class]: end_nodes_by_class[send_class]])
            #print()

            #print("receive_counts==0", receive_counts)
            #print("classes", classes)
            num_active_classes = 0
            #if (end_nodes_by_class[send_class]-start_nodes_by_class[send_class]) == 1:
            #    single_node_propagation(send_class, nodes_by_class, start_nodes_by_class, end_neighbors, neighbors,
            #                   receive_counts, classes, active_classes, position_of_node, class_in_queue, queue, num_classes,
            #                            queue_R, received_nodes_by_class)
            #    continue
            #print("max", active_nodes_in_class[:num_classes].max())
            if (start_class-end_class) == 1:
                # print("special")
                # special_case +=1
                # special case if the class has only one(!) node
                sending_node = nodes_by_class[start_class]
                #print("lonely_node", sending_node)
                num_active_classes = 0
                for j in range(start_neighbors[sending_node], end_neighbors[sending_node]):
                    neigh = neighbors[j]
                    #num_messages+=1
                    if not node_is_active[neigh]:
                        neigh_class = classes[neigh]
                        # mark node as needing processing
                        received_nodes_by_class[start_nodes_by_class[neigh_class] + active_nodes_in_class[neigh_class]] = neigh
                        active_nodes_in_class[neigh_class] +=1

                        if active_nodes_in_class[neigh_class] == 1:
                            # mark class as active
                            active_classes[num_active_classes] = neigh_class
                            num_active_classes+=1
                #if num_active_classes == 1:
                #    num_active_1 +=1
                #elif num_active_classes == 2:
                #    num_active_2 +=1
                for active_class_index in range(num_active_classes):
                    active_class = active_classes[active_class_index]
                    total_nodes_this_class = end_nodes_by_class[active_class] - start_nodes_by_class[active_class]

                    if total_nodes_this_class==1:
                        # classes with only 1 node never need to be a receiving class again
                        #  thus set the node to be active
                        i_node = start_nodes_by_class[active_class]
                        node = received_nodes_by_class[i_node]
                        node_is_active[node] = True
                        continue

                    if total_nodes_this_class == active_nodes_in_class[active_class]:
                        active_nodes_in_class[active_class] = 0
                        continue
                    else:
                        # prepare the two new classes
                        L = start_nodes_by_class[active_class]
                        R = start_nodes_by_class[active_class] + active_nodes_in_class[active_class]
                        active_nodes_in_class[active_class] = 0
                        new_class = num_classes
                        num_classes+=1
                        start_nodes_by_class[new_class] = L
                        end_nodes_by_class[new_class] = R

                        out_classes[new_class,0] = L
                        out_classes[new_class,1] = R
                        out_classes[new_class,2] = depth
                        start_nodes_by_class[active_class] = R
                        #print("start", start_nodes_by_class[:num_classes])
                        #print("end", end_nodes_by_class[:num_classes])
                        index = L
                        for i_node in range(L, R):
                            node = received_nodes_by_class[i_node]
                            classes[node] = new_class
                            # swap node positions
                            start_pos_node = position_of_node[node]
                            swap_node = nodes_by_class[index]
                            nodes_by_class[index] = node
                            position_of_node[node] = index
                            nodes_by_class[start_pos_node] = swap_node
                            position_of_node[swap_node] = start_pos_node
                            #nodes_indices_by_group[group_id]+=1
                            index+=1

                        #print("in queue A", active_class, class_in_queue[active_class])
                        # if class_in_queue[active_class]:
                        #     put_in = new_class
                        # else:
                        if active_nodes_in_class[active_class] > total_nodes_this_class//2:
                            put_in=new_class
                        else:
                            put_in=active_class

                        queue[queue_R,0] = start_nodes_by_class[put_in]
                        queue[queue_R,1] = end_nodes_by_class[put_in]
                        #class_in_queue[put_in] = True
                        queue_R+=1
                #print(classes)
                #classes_are_mono(start_nodes_by_class, end_nodes_by_class, num_classes, nodes_by_class, classes)
                continue

            # normal_case +=1


            for i in range(start_class, end_class):
                sending_node = nodes_by_class[i]
                for j in range(start_neighbors[sending_node], end_neighbors[sending_node]):
                    neigh = neighbors[j]
                    # print("sending", sending_node, neigh)
                    # num_messages+=1
            #for sending_node in nodes_by_class[start_nodes_by_class[send_class]: end_nodes_by_class[send_class]]:
                #for neigh in neighbors[end_neighbors[sending_node]:end_neighbors[sending_node+1]]:
                    receive_counts[neigh] += 1
                    neigh_class = classes[neigh]
                    if not node_is_active[neigh]:
                        # mark node as needing processing
                        received_nodes_by_class[start_nodes_by_class[neigh_class] + active_nodes_in_class[neigh_class]] = neigh
                        active_nodes_in_class[neigh_class] +=1
                        node_is_active[neigh] = True

                        if active_nodes_in_class[neigh_class] == 1: # the current neigh node makes it's class active
                            # enque class into queue of active classes
                            active_classes[num_active_classes] = neigh_class
                            num_active_classes+=1
            #for class_ in active_classes[:num_active_classes]:
            #    active_nodes_in_class[class_] = 0
            #print()
            #print()
            #print("active classes")
            #print(num_active_classes)
            #print("active_classes", active_classes[:num_active_classes])
            #print("count", receive_counts)
            #print("-------------------------------------------------------")
            #print("ordered", classes[nodes_by_class])
            # print("receive", receive_counts)
            for active_class_index in range(num_active_classes): # loop over all classes which were potentially split by this action
                #classes_are_mono(start_nodes_by_class, end_nodes_by_class, num_classes, nodes_by_class, classes)
                active_class = active_classes[active_class_index]
                #print("class ranges", list( zip(start_nodes_by_class[:num_classes], end_nodes_by_class[:num_classes])))
                #s=[]
                #for start, end in zip(start_nodes_by_class[:num_classes], end_nodes_by_class[:num_classes]):
                #    s.append("".join(str(classes[node]) for node in nodes_by_class[start: end]))


                #print(" | ".join(s))
                #print(active_classes[:num_active_classes])

                #print("active class", active_class)
                #print("active range", start_nodes_by_class[active_class], end_nodes_by_class[active_class])
                num_active_nodes_this_class = active_nodes_in_class[active_class]


                # resetting node information
                for i_node in range(start_nodes_by_class[active_class], start_nodes_by_class[active_class] + num_active_nodes_this_class):
                    node = received_nodes_by_class[i_node]
                    node_is_active[node]=False
                #print("activ", num_active_nodes_this_class)
                active_nodes_in_class[active_class] = 0

                total_nodes_this_class = end_nodes_by_class[active_class] - start_nodes_by_class[active_class]
                #print("total", total_nodes_this_class)
                #print()
                #assert total_nodes_this_class >= num_active_nodes_this_class
                if total_nodes_this_class==1:
                    # classes with only 1 node never need to be an receiving class again
                    i_node = start_nodes_by_class[active_class]
                    node = received_nodes_by_class[i_node]
                    node_is_active[node] = True # this ensures this node never lands in the active queue again
                    continue




                #print(total_nodes_this_class, non_active_group_size)
                num_groups = 0
                # find the groups and the number of nodes per group
                for i_node in range(start_nodes_by_class[active_class], start_nodes_by_class[active_class] + num_active_nodes_this_class):
                    node = received_nodes_by_class[i_node]
                    group_id = receive_counts[node]
                    num_nodes_per_group_scattered[group_id]+= 1 # identify group sizes
                    if num_nodes_per_group_scattered[group_id] == 1: #add this group to queue of groups
                        group_ids[num_groups] = group_id
                        num_groups += 1
                #print("num_g", num_groups)

                # there might be nodes which are not adjacent to the currently sending class
                #    we are treating these incoming degree zero nodes here
                non_active_group_size = total_nodes_this_class - num_active_nodes_this_class
                if non_active_group_size > 0:
                    group_ids[num_groups] = 0
                    num_nodes_per_group_scattered[0] = non_active_group_size
                    num_groups+=1
                #print("groups",group_ids[:num_groups])
                #print("ngrou", num_groups)

                # if num_groups==1:
                #     num_groups_1+=1
                # elif num_groups==2:
                #     num_groups_2+=1
                # elif num_groups==3:
                #     num_groups_3+=1
                # else:
                #     num_groups_x+=1


                if num_groups==1: # nothing to be done, active class is not split
                    # reset the counts
                    for node in nodes_by_class[start_nodes_by_class[active_class]: end_nodes_by_class[active_class]]:
                        receive_counts[node]=0
                    for i in range(num_groups):
                        num_nodes_per_group_scattered[group_ids[i]] = 0
                    continue

                # min_group_id = group_ids[:num_groups].min()
                # if min_group_id == 0:
                #     min_group_id_0 +=1
                # elif min_group_id == 1:
                #     min_group_id_1 +=1
                # else:
                #     min_group_id_x +=1

                # max_group_id = group_ids[:num_groups].max()
                # if max_group_id == 1:
                #     max_group_id_1 +=1
                # elif max_group_id == 2:
                #     max_group_id_2 +=1
                # else:
                #     max_group_id_x+=1
                # if min_group_id == 0 and max_group_id == 1:
                #     min_max_group_id_0_1 +=1


                # collect num_nodes_per_group from scattered
                for i in range(num_groups):
                    num_nodes_per_group[i] = num_nodes_per_group_scattered[group_ids[i]]
                    num_nodes_per_group_scattered[group_ids[i]] = 0
                #print(num_nodes_per_group)

                # in the following we determine two special classes: 1) the not_relabeled_group and 2) the not_enqueued aka the largest_group
                #     the not relabeled group is usually the degree zero node group, but in case there is no degree zero group, take the largest
                #     the largest group is simply one of the largest groups (doesn't matter which)
                # collect largest group statistics
                _largest_group_index = np.argmax(num_nodes_per_group[:num_groups])
                largest_group_size = num_nodes_per_group[_largest_group_index]
                if largest_group_size == non_active_group_size: # in case that the largest and the zero partition are of identical size, just take 0
                    largest_group_id = 0
                else:
                    largest_group_id = group_ids[_largest_group_index]
                if non_active_group_size == 0:
                    not_relabeled_group_id = largest_group_id
                else:
                    not_relabeled_group_id = 0
                # if largest_group_id == 0:
                #     largest_group_id_0 +=1

                # ----- begin collecting nodes into groups -----
                # cumsum num_nodes_per_group
                for i in range(1, num_groups):
                    num_nodes_per_group[i] = num_nodes_per_group[i-1] + num_nodes_per_group[i]
                #print("largest", largest_group_id)
                #print("start", start_nodes_by_class[:num_classes+1])
                #print("end  ", end_nodes_by_class[:num_classes+1])
                #print("per_g", num_nodes_per_group[:num_groups])
                # scatter node indices to group locations
                offset = start_nodes_by_class[active_class]
                end_offset = start_nodes_by_class[active_class] + num_active_nodes_this_class
                assert offset < end_offset
                #print("groups", group_ids[:num_groups])
                for i in range(num_groups):
                    group_id = group_ids[i]
                    if i == 0:
                        nodes_indices_by_group[group_id] = 0 # nodes_indices_by_group will contain the position of this group in the order
                    else:
                        nodes_indices_by_group[group_id] = num_nodes_per_group[i-1]

                    if group_id == not_relabeled_group_id: # there are some nodes that are not relabeled, they keep the active class
                        class_name_for_group[group_id] = active_class
                    else:
                        class_name_for_group[group_id] = num_classes # this group will become a new class
                        num_classes+=1
                    #print("in queue B", active_class, class_in_queue[active_class])
                    put_in = False
                    # if class_in_queue[active_class]:
                    #     if group_id != not_relabeled_group_id:
                    #         put_in = True
                    # else:
                    if group_id == largest_group_id:
                        put_in=False
                    else:
                        put_in=True
                    group_class = class_name_for_group[group_id]
                    start_nodes_by_class[group_class] = offset + nodes_indices_by_group[group_id]

                    # if i == 0:
                    #     start_nodes_by_class[group_class] = offset
                    # else:
                    #     start_nodes_by_class[group_class] = offset + num_nodes_per_group[i-1]
                    end_nodes_by_class[group_class] = offset + num_nodes_per_group[i]

                    if put_in:
                        queue[queue_R,0] = start_nodes_by_class[group_class]
                        queue[queue_R,1] = end_nodes_by_class[group_class]
                        # class_in_queue[group_class] = True
                        queue_R+=1

                    if group_class != active_class: # active class is already present in out_classes
                        out_classes[group_class,0] = start_nodes_by_class[group_class]
                        out_classes[group_class,1] = end_nodes_by_class[group_class]
                        out_classes[group_class,2] = depth


                    #print("changing", new_group_ids[group_id])
                    #assert end_nodes_by_class[class_name_for_group[group_id]]>start_nodes_by_class[class_name_for_group[group_id]]

                #print(nodes_indices_by_group[group_ids[:num_groups]])
                #print(class_name_for_group[group_ids[:num_groups]])
                #print("start", start_nodes_by_class[:num_classes+1])
                #print("end  ", end_nodes_by_class[:num_classes+1])
                #print(nodes_by_class)
                for i_node in range(offset, end_offset):
                    node = received_nodes_by_class[i_node]
                    #print("processing", node, classes[node], "->",  class_name_for_group[group_id])
                    group_id = receive_counts[node]
                    receive_counts[node] = 0 # reset this value, to be used again in the future
                    #print(group_id)
                    classes[node] = class_name_for_group[group_id]
                    group_class = class_name_for_group[group_id]

                    #if (end_nodes_by_class[group_class]-start_nodes_by_class[group_class]) == 1:
                    #if True:
                        #print(neighbors)
                        #print(start_neighbors)
                        #print(end_neighbors)
                    #    for i_remove in range(in_degree[node], in_degree[node+1]):
                    #        to_remove_pos = in_neighbors_position[i_remove]
                    #        affected_node = in_neighbors[i_remove]

                            # def remove_from_out(to_delete, values, original, where, end_neighbors, node)
                            #print("affected", affected_node, to_remove_pos)
                    #        remove_from_out(to_remove_pos, neighbors, original, where, end_neighbors, affected_node)

                    #       node_is_active[node] = True
                        #print(neighbors, end_neighbors)

                    if group_id == not_relabeled_group_id:
                        continue

                    # swap node positions
                    start_pos_node = position_of_node[node]
                    target_index = nodes_indices_by_group[group_id]+offset # target location of node
                    swap_node = nodes_by_class[target_index] # the node currently at the swap position
                    nodes_by_class[target_index] = node  # place current node
                    position_of_node[node] = target_index  # set position of current node
                    nodes_by_class[start_pos_node] = swap_node # place other node
                    position_of_node[swap_node] = start_pos_node # set position of other node
                    nodes_indices_by_group[group_id]+=1 # increase counter
                #classes_are_mono(start_nodes_by_class, end_nodes_by_class, num_classes, nodes_by_class, classes)
                    #print("swap", node, swap_node, start_pos_node, index)
            #print(nodes_by_class)
            #print(nodes_by_class[position_of_node])

    if starts_from_degree:
        out_classes[0,0] = 0
        out_classes[0,1] = n_nodes
        out_classes[0,2] = 0

    # print("num_groups1", num_groups_1)
    # print("num_groups2", num_groups_2)
    # print("num_groups3", num_groups_3)
    # print("num_groupsx", num_groups_x)

    # print("min_group_id_0", min_group_id_0)
    # print("min_group_id_1", min_group_id_1)
    # print("min_group_id_x", min_group_id_x)
    # print("max_group_id_1", max_group_id_1)
    # print("max_group_id_2", max_group_id_2)
    # print("max_group_id_x", max_group_id_x)
    # print("minmax_group_id_0_1", min_max_group_id_0_1)
    # print("largest_group_id_0", largest_group_id_0)
    # print("special_case", special_case)
    # print("normal_case ", normal_case)
    # print("neigh_10", neighborhood_10)
    # print("neigh_X ", neighborhood_x)
    # print("all_unqiue", all_unique)
    # print("num_active_1", num_active_1)
    # print("num_active_2", num_active_2)
    # print("total_  messages", num_messages)
    # print("classes_processed", classes_processed)
    # start_nodes_by_class[:num_classes].sort()
    # end_nodes_by_class[:num_classes].sort()
    # print(start_nodes_by_class[:10])
    # print(end_nodes_by_class[:10])
    # print(out_classes[:20])

    #print()
    #print("result")
    #print(classes)
    #print(nodes_by_class)
    return position_of_node, out_classes[0:num_classes,:]
