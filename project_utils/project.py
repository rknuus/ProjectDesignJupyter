#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
from datetime import date, timedelta
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random
import math
import numpy as np
import json
from matplotlib.widgets import Button


class Project:
    def __init__(self, name: str, start_date: date, tasks: list[dict] | None = None):
        self.name = name
        self.start_date = start_date
        self.next_task_id = 1
        self.tasks = []
        for task in tasks or []:
            self.add_task(task["Name"], task["Duration"], task["Predecessors"])
        print("ready")

    def add_task(self, name: str, duration_days: int, predecessors: list[int] | None = None):
        if duration_days % 5 != 0:
            raise ValueError("Duration must be a multiple of 5 days.")
        task = {
            "ID": self.next_task_id,
            "Name": name,
            "Duration": duration_days,
            "Predecessors": sorted(set(predecessors)) if predecessors else []
        }
        self.tasks.append(task)
        self.next_task_id += 1
        return task

    def get_task_table(self):
        return pd.DataFrame(self.tasks)

    def align_to_monday(self, d: date):
        return d + timedelta(days=(7 - d.weekday()) % 7)
    
    def calendar_days_to_working_days(self, calendar_days: int) -> int:
        """Convert calendar days since project start to working days (5-day weeks)."""
        if calendar_days < 0:
            return 0
        # Each complete week has 5 working days
        complete_weeks = calendar_days // 7
        remaining_days = calendar_days % 7
        
        # Count working days in the remaining partial week
        # Assuming project starts on Monday (weekday 0)
        start_weekday = self.start_date.weekday()
        working_days_in_remainder = 0
        for i in range(remaining_days):
            day_of_week = (start_weekday + i) % 7
            if day_of_week < 5:  # Monday=0, Friday=4
                working_days_in_remainder += 1
        
        return complete_weeks * 5 + working_days_in_remainder
    
    def working_days_to_calendar_days(self, working_days: int) -> int:
        """Convert working days to calendar days since project start."""
        if working_days <= 0:
            return 0
        # Each complete work week (5 working days) = 7 calendar days
        complete_work_weeks = working_days // 5
        remaining_working_days = working_days % 5
        
        calendar_days = complete_work_weeks * 7
        
        # Add calendar days for remaining working days
        start_weekday = self.start_date.weekday()
        days_added = 0
        working_days_added = 0
        
        while working_days_added < remaining_working_days:
            day_of_week = (start_weekday + calendar_days + days_added) % 7
            if day_of_week < 5:  # Working day
                working_days_added += 1
            days_added += 1
        
        return calendar_days + days_added

    def schedule_tasks(self, start_date: date | None = None):
        if start_date is not None:
            self.start_date = start_date
        # Index tasks by ID for quick lookup
        id_to_task = {t["ID"]: t for t in self.tasks}
        scheduled = {}
        scheduled_days = {}  # Track days since start for calculations

        # Forward pass: Calculate EST and EFTe in working days since project start
        def compute_early_dates(task_id):
            if task_id in scheduled:
                return scheduled[task_id]

            task = id_to_task[task_id]
            if not task["Predecessors"]:
                start = self.start_date
                start_working_day = 0  # Project starts at working day 0
            else:
                # Get finish working days of all predecessors
                pred_results = [compute_early_dates(pid) for pid in task["Predecessors"]]
                pred_finish_working_days = [scheduled_days[pid][1] for pid in task["Predecessors"]]
                latest_finish_working_day = max(pred_finish_working_days)
                
                # Next task starts on the next working day
                start_working_day = latest_finish_working_day
                
                # Convert to calendar date for actual scheduling
                start_calendar_days = self.working_days_to_calendar_days(start_working_day)
                start = self.start_date + timedelta(days=start_calendar_days)
                start = self.align_to_monday(start)

            # Calculate finish in working days
            finish_working_day = start_working_day + task["Duration"]
            
            # Convert to calendar date for actual scheduling
            finish_calendar_days = self.working_days_to_calendar_days(finish_working_day)
            finish = self.start_date + timedelta(days=finish_calendar_days - 1)
            
            scheduled[task_id] = (start, finish)
            scheduled_days[task_id] = (start_working_day, finish_working_day)
            return (start, finish)

        # Run forward pass for all tasks
        for task in self.tasks:
            compute_early_dates(task["ID"])

        # Project finish = latest of all early_finish working days
        project_finish_working_day = max(finish_working_day for _, finish_working_day in scheduled_days.values())
        project_finish = max(finish for _, finish in scheduled.values())

        # Build successor map for backward pass
        successors = {t["ID"]: [] for t in self.tasks}
        for task in self.tasks:
            for pred in task["Predecessors"]:
                successors[pred].append(task["ID"])

        # Backward pass: Calculate LST and LFTe in working days since project start
        latest = {}
        latest_working_days = {}

        def compute_latest_dates(task_id):
            if task_id in latest:
                return latest[task_id]

            task = id_to_task[task_id]

            # For tasks with no predecessors, they should be on the critical path
            if not task["Predecessors"]:
                early_start, early_finish = scheduled[task_id]
                early_start_working_day, early_finish_working_day = scheduled_days[task_id]
                latest[task_id] = (early_start, early_finish)
                latest_working_days[task_id] = (early_start_working_day, early_finish_working_day)
                return latest[task_id]

            if not successors[task_id]:
                # No successors means this is an end task
                late_finish = project_finish
                late_finish_working_day = project_finish_working_day
            else:
                # Latest finish = earliest of successor late starts (in working days)
                succ_results = [compute_latest_dates(sid) for sid in successors[task_id]]
                succ_starts = [result[0] for result in succ_results]
                late_finish = min(succ_starts) - timedelta(days=1)
                late_finish_working_day = min(latest_working_days[sid][0] for sid in successors[task_id])

            duration = task["Duration"]
            late_start_working_day = late_finish_working_day - duration

            # Convert to calendar date for actual scheduling
            late_start_calendar_days = self.working_days_to_calendar_days(late_start_working_day)
            late_start_calc = self.start_date + timedelta(days=late_start_calendar_days)
            late_start = self.align_to_previous_monday(late_start_calc)

            # Convert back to working days and recalculate finish
            late_start_calendar_days_aligned = (late_start - self.start_date).days
            late_start_working_day = self.calendar_days_to_working_days(late_start_calendar_days_aligned)
            late_finish_working_day = late_start_working_day + duration

            # Calculate late finish calendar date
            late_finish_calendar_days = self.working_days_to_calendar_days(late_finish_working_day)
            late_finish = self.start_date + timedelta(days=late_finish_calendar_days - 1)

            latest[task_id] = (late_start, late_finish)
            latest_working_days[task_id] = (late_start_working_day, late_finish_working_day)
            return latest[task_id]

        # Run backward pass for all tasks
        for task in reversed(self.tasks):
            compute_latest_dates(task["ID"])

        # Calculate additional project management metrics
        data = []
        for task in self.tasks:
            start, finish = scheduled[task["ID"]]  # Calendar dates
            late_start, late_finish = latest[task["ID"]]  # Calendar dates
            start_working_day, finish_working_day = scheduled_days[task["ID"]]  # Working days
            late_start_working_day, late_finish_working_day = latest_working_days[task["ID"]]  # Working days
            
            # EST (Earliest Start Time): Already calculated in forward pass
            est = start_working_day
            
            # EFT (Earliest Finish Time): Already calculated in forward pass  
            eft = finish_working_day
            
            # LST (Latest Start Time): Already calculated in backward pass
            lst = late_start_working_day
            
            # LFT (Latest Finish Time): Already calculated in backward pass
            lft = late_finish_working_day
            
            # TF (Total Float): LFT - EFT
            tf = lft - eft
            
            # FF (Free Float): min(EST of subsequent activities) - EFT
            if not successors[task["ID"]]:
                # No successors means free float = total float
                ff = tf
            else:
                successor_est_values = [scheduled_days[sid][0] for sid in successors[task["ID"]]]
                min_successor_est = min(successor_est_values)
                ff = min_successor_est - eft
            
            # IF (Interfering Float): TF - FF
            if_float = tf - ff
            
            data.append({
                "ID": task["ID"],
                "Name": task["Name"],
                "Duration": task["Duration"],
                "Start": start,
                "Finish": finish,
                "EST": est,  # Earliest Start Time (working days)
                "EFT": eft,  # Earliest Finish Time (working days)
                "LST": lst,  # Latest Start Time (working days)
                "LFT": lft,  # Latest Finish Time (working days)
                "TF": tf,    # Total Float (working days)
                "FF": ff,    # Free Float (working days)
                "IF": if_float,  # Interfering Float (working days)
                "EFTe": finish_working_day,  # Earliest Finish Time as working days since start
                "Latest Start": late_start,
                "Latest Finish": late_finish,
                "LFTe": late_finish_working_day,  # Latest Finish Time as working days since start
                "Predecessors": task["Predecessors"]
            })

        return pd.DataFrame(data).sort_values("ID")

    def align_to_previous_monday(self, d: date):
        """Align date to the previous Monday (or same day if already Monday)"""
        return d - timedelta(days=d.weekday())

    def calculate_working_day_float(self, early_start: date, late_start: date):
        """Calculate float in working days only (excludes weekends)"""
        if early_start == late_start:
            return 0

        # Count complete weeks between the dates
        weeks_diff = (late_start - early_start).days // 7
        working_day_float = weeks_diff * 5

        return working_day_float

    def schedule_tasks_with_float(self):
        """Schedule tasks with float calculation using both forward and backward pass"""
        id_to_task = {t["ID"]: t for t in self.tasks}
        scheduled = {}

        # Forward pass
        def compute_early_dates(task_id):
            if task_id in scheduled:
                return scheduled[task_id]

            task = id_to_task[task_id]
            if not task["Predecessors"]:
                early_start = self.start_date
            else:
                pred_ends = [compute_early_dates(pid)[1] for pid in task["Predecessors"]]
                latest_pred_finish = max(pred_ends)
                early_start = self.align_to_monday(latest_pred_finish + timedelta(days=1))

            early_finish = early_start + timedelta(days=task["Duration"] - 1)
            scheduled[task_id] = (early_start, early_finish)
            return scheduled[task_id]

        for task in self.tasks:
            compute_early_dates(task["ID"])

        # Project finish = latest of all early_finish
        project_finish = max(finish for _, finish in scheduled.values())

        # Build successor map
        successors = {t["ID"]: [] for t in self.tasks}
        for task in self.tasks:
            for pred in task["Predecessors"]:
                successors[pred].append(task["ID"])

        # Backward pass: compute latest finish/start
        latest = {}

        def compute_latest_dates(task_id):
            if task_id in latest:
                return latest[task_id]

            task = id_to_task[task_id]

            # Special case first: For tasks with no predecessors (first tasks),
            # they should be on the critical path and have zero float
            if not task["Predecessors"]:
                early_start, early_finish = scheduled[task_id]
                latest[task_id] = (early_start, early_finish)
                return latest[task_id]

            if not successors[task_id]:
                late_finish = project_finish
            else:
                succ_starts = [compute_latest_dates(sid)[0] for sid in successors[task_id]]
                late_finish = min(succ_starts) - timedelta(days=1)

            duration = task["Duration"]
            late_start_calc = late_finish - timedelta(days=duration - 1)

            # Apply Monday alignment to latest start
            late_start = self.align_to_previous_monday(late_start_calc)

            # Recalculate latest finish based on aligned start
            late_finish = late_start + timedelta(days=duration - 1)

            latest[task_id] = (late_start, late_finish)
            return latest[task_id]

        for task in reversed(self.tasks):  # safe because it's a DAG
            compute_latest_dates(task["ID"])

        # Combine all into a table
        result = []
        for task in self.tasks:
            est, eft = scheduled[task["ID"]]
            lst, lft = latest[task["ID"]]
            calendar_float = (lst - est).days
            working_day_float = self.calculate_working_day_float(est, lst)

            result.append({
                "ID": task["ID"],
                "Name": task["Name"],
                "Duration": task["Duration"],
                "Start": est,
                "Finish": eft,
                "Latest Start": lst,
                "Latest Finish": lft,
                "Calendar Float": calendar_float,
                "Working Day Float": working_day_float,
                "Predecessors": task["Predecessors"]
            })

        return pd.DataFrame(result).sort_values("ID")

    def get_criticality_category(self, float_value, thresholds=None):
        """
        Determine criticality category based on float value.

        Args:
            float_value: The float value in days
            thresholds: Dict with 'near_critical', 'medium_critical' thresholds

        Returns:
            tuple: (category_name, color, is_bold)
        """
        if thresholds is None:
            thresholds = {
                'near_critical': 5,    # 1-5 days float
                'medium_critical': 15  # 6-15 days float
            }

        if float_value == 0:
            return ('critical', 'black', True)
        elif 1 <= float_value <= thresholds['near_critical']:
            return ('near_critical', 'red', False)
        elif float_value <= thresholds['medium_critical']:
            return ('medium_critical', 'orange', False)
        else:
            return ('uncritical', 'green', False)

    def create_topological_layout(self, G):
        """
        Create a strict left-to-right topological layout ensuring arrows never go backwards.
        """
        # Step 1: Perform topological sort to get proper ordering
        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXError:
            # If there are cycles, fall back to node ordering
            topo_order = sorted(G.nodes())

        # Step 2: Assign levels (x-coordinates) based on longest path from start
        levels = {}

        # Initialize nodes with no predecessors at level 0
        for node in G.nodes():
            if G.in_degree(node) == 0:
                levels[node] = 0

        # Assign levels based on longest path from any predecessor
        for node in topo_order:
            if node not in levels:
                # Find maximum level of all predecessors and add 1
                pred_levels = [levels.get(pred, 0) for pred in G.predecessors(node)]
                levels[node] = max(pred_levels, default=0) + 1

        # Step 3: Group nodes by level for vertical distribution
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)

        # Step 4: Create positions ensuring left-to-right flow
        pos = {}
        max_level = max(levels.values()) if levels else 0

        # Horizontal spacing - ensure good separation between levels
        horizontal_spacing = 2.0
        vertical_spacing = 1.5

        for level, nodes in level_groups.items():
            # X-coordinate: strictly increasing from left to right
            x = level * horizontal_spacing

            # Y-coordinates: distribute nodes vertically within the level
            num_nodes = len(nodes)
            if num_nodes == 1:
                y = 0  # Center single nodes
                pos[nodes[0]] = (x, y)
            else:
                # Sort nodes for consistent positioning
                sorted_nodes = sorted(nodes)
                for i, node in enumerate(sorted_nodes):
                    # Center the group and spread vertically
                    y_offset = (i - (num_nodes - 1) / 2) * vertical_spacing
                    pos[node] = (x, y_offset)

        return pos, levels

    def map_aoa_activities_to_floats(self, G, task_floats):
        """
        Map AoA network activities to their float values.
        For AoA networks, activities are edges, not nodes.
        """
        activity_floats = {}

        for u, v, data in G.edges(data=True):
            if not data.get('is_dummy', False):
                activity_id = data.get('activity_id')
                if activity_id in task_floats:
                    activity_floats[(u, v)] = task_floats[activity_id]
                else:
                    activity_floats[(u, v)] = 0  # Default for missing data
            else:
                # Dummy activities inherit float from their "parent" relationship
                activity_floats[(u, v)] = 0  # For now, treat as critical

        return activity_floats

    def create_float_based_layout(self, G):
        """
        Create layout where:
        - X-axis: Topological level (left-to-right flow)
        - Y-axis: Float value (critical path centered, higher float towards edges)
        """
        print("Creating float-based layout...")

        # Step 1: Get float values for all tasks
        task_floats = self.get_task_floats()
        print(f"Float values: {task_floats}")

        # Step 2: Create basic topological layout for X-coordinates
        pos, levels = self.create_topological_layout(G)

        # Step 3: Map activities to their float values
        activity_floats = self.map_aoa_activities_to_floats(G, task_floats)

        # Step 4: Calculate float values for events (nodes)
        # Strategy: Each event gets the minimum float of activities that END at that event
        event_floats = {}

        for node in G.nodes():
            # Find all activities that end at this node
            ending_activities = []
            for u, v, data in G.edges(data=True):
                if v == node and not data.get('is_dummy', False):
                    ending_activities.append((u, v))

            if ending_activities:
                # Use minimum float of ending activities (most critical determines position)
                node_float = min(activity_floats.get(activity, 0) for activity in ending_activities)
            else:
                # No activities end here (start node), use minimum float of outgoing activities
                outgoing_activities = []
                for u, v, data in G.edges(data=True):
                    if u == node and not data.get('is_dummy', False):
                        outgoing_activities.append((u, v))

                if outgoing_activities:
                    node_float = min(activity_floats.get(activity, 0) for activity in outgoing_activities)
                else:
                    node_float = 0  # Default

            event_floats[node] = node_float

        print(f"Event floats: {event_floats}")

        # Step 5: Apply float-based vertical positioning
        float_based_pos = {}

        # Get float range for scaling
        all_floats = list(event_floats.values())
        min_float = min(all_floats)
        max_float = max(all_floats)
        float_range = max_float - min_float if max_float > min_float else 1

        print(f"Float range: {min_float} to {max_float}")

        # Layout parameters
        vertical_spread = 4.0  # Maximum vertical distance from center

        for node in G.nodes():
            x = pos[node][0]  # Keep X-coordinate from topological layout
            node_float = event_floats[node]

            # Map float to Y-coordinate
            if float_range > 0:
                # Normalize float to [-1, 1] range (0 float = 0, max float = ±1)
                normalized_float = (node_float - min_float) / float_range
                # Convert to signed value: 0 float -> center, higher float -> edges
                # Use alternating pattern to distribute high-float activities above and below
                if normalized_float == 0:
                    y = 0  # Critical path at center
                else:
                    # Alternate high-float activities above/below center
                    sign = 1 if (node % 2 == 0) else -1  # Alternate by node ID
                    y = sign * normalized_float * vertical_spread
            else:
                y = 0  # All same float, put at center

            float_based_pos[node] = (x, y)

        # Step 6: Collision detection and resolution
        node_resolved_pos = self.resolve_node_collisions(float_based_pos, event_floats)

        # Step 7: Edge/Activity overlap resolution
        overlap_resolved_pos = self.resolve_activity_overlaps(G, node_resolved_pos, event_floats)
        
        # Step 8: Crossing minimization
        final_pos = self.minimize_arrow_crossings(G, overlap_resolved_pos, event_floats)

        return final_pos, event_floats

    def get_task_floats(self):
        """
        Calculate float values for all tasks using the Project class method.
        Returns a dictionary mapping task_id -> working_day_float
        """
        # Use the project's scheduling method to get float data
        task_df = self.schedule_tasks_with_float()

        # Extract float values into a dictionary
        float_dict = {}
        for _, row in task_df.iterrows():
            float_dict[row['ID']] = row['Working Day Float']

        return float_dict

    def resolve_node_collisions(self, pos, event_floats, min_separation=0.8):
        """
        Detect and resolve node collisions by adjusting vertical positions.
        Keeps critical path nodes as close to center as possible.
        """
        print("Resolving node collisions...")

        # Group nodes by X-coordinate (topological level)
        level_groups = {}
        for node, (x, y) in pos.items():
            if x not in level_groups:
                level_groups[x] = []
            level_groups[x].append((node, y))

        adjusted_pos = pos.copy()

        for x_level, nodes_at_level in level_groups.items():
            if len(nodes_at_level) <= 1:
                continue  # No collisions possible

            # Sort nodes by their current Y position
            nodes_at_level.sort(key=lambda item: item[1])

            # Check for collisions and resolve
            collision_groups = []
            current_group = [nodes_at_level[0]]

            for i in range(1, len(nodes_at_level)):
                prev_node, prev_y = nodes_at_level[i-1]
                curr_node, curr_y = nodes_at_level[i]

                # Check if nodes are too close vertically
                if abs(curr_y - prev_y) < min_separation:
                    # Add to current collision group
                    current_group.append((curr_node, curr_y))
                else:
                    # Start new group
                    if len(current_group) > 1:
                        collision_groups.append(current_group)
                    current_group = [(curr_node, curr_y)]

            # Don't forget the last group
            if len(current_group) > 1:
                collision_groups.append(current_group)

            # Resolve each collision group
            for group in collision_groups:
                resolved_positions = self.resolve_collision_group(group, event_floats, min_separation)

                # Update positions
                for node, new_y in resolved_positions:
                    adjusted_pos[node] = (adjusted_pos[node][0], new_y)

        return adjusted_pos

    def resolve_collision_group(self, collision_group, event_floats, min_separation):
        """
        Resolve collisions within a group of overlapping nodes.
        Strategy: Keep critical path (0 float) nodes closest to center.
        """
        # Sort by criticality (float value), then by node ID for consistency
        sorted_group = sorted(collision_group, key=lambda item: (event_floats[item[0]], item[0]))

        # Find the center Y position of the group
        center_y = sum(y for _, y in collision_group) / len(collision_group)

        resolved = []
        num_nodes = len(sorted_group)

        if num_nodes == 2:
            # Simple case: two nodes
            node1, _ = sorted_group[0]
            node2, _ = sorted_group[1]

            # Place more critical node closer to center
            float1 = event_floats[node1]
            float2 = event_floats[node2]

            if float1 <= float2:
                # node1 is more critical, place closer to center
                y1 = center_y - min_separation / 2
                y2 = center_y + min_separation / 2
            else:
                # node2 is more critical, place closer to center
                y1 = center_y + min_separation / 2
                y2 = center_y - min_separation / 2

            resolved = [(node1, y1), (node2, y2)]

        else:
            # Multiple nodes: distribute around center, keeping critical nodes central
            total_height = (num_nodes - 1) * min_separation
            start_y = center_y - total_height / 2

            for i, (node, _) in enumerate(sorted_group):
                new_y = start_y + i * min_separation
                resolved.append((node, new_y))

        print(f"  Resolved collision: {len(collision_group)} nodes at level, spread over {min_separation * (num_nodes-1):.1f} units")

        return resolved

    def resolve_activity_overlaps(self, G, pos, event_floats, min_arrow_separation=0.6):
        """
        Detect and resolve overlapping activities (edges/arrows) by adjusting node positions.
        """
        print("Resolving activity overlaps...")

        adjusted_pos = pos.copy()

        # Get all real activities (non-dummy edges)
        real_activities = []
        for u, v, data in G.edges(data=True):
            if not data.get('is_dummy', False):
                real_activities.append((u, v, data))

        # Find overlapping activity pairs
        overlapping_pairs = []

        for i in range(len(real_activities)):
            for j in range(i + 1, len(real_activities)):
                u1, v1, data1 = real_activities[i]
                u2, v2, data2 = real_activities[j]

                # Check if arrows are too close (similar paths)
                if self.are_arrows_overlapping(u1, v1, u2, v2, pos, min_arrow_separation):
                    activity1_id = data1.get('activity_id', 'unknown')
                    activity2_id = data2.get('activity_id', 'unknown')
                    overlapping_pairs.append(((u1, v1, activity1_id), (u2, v2, activity2_id)))

        # Resolve each overlapping pair
        for (u1, v1, id1), (u2, v2, id2) in overlapping_pairs:
            print(f"  Resolving overlap: Activity {id1} (E{u1}→E{v1}) vs Activity {id2} (E{u2}→E{v2})")

            # Determine which activity is more critical (lower float)
            float1 = event_floats.get(u1, 0) + event_floats.get(v1, 0)  # Combined float of start+end
            float2 = event_floats.get(u2, 0) + event_floats.get(v2, 0)

            if float1 <= float2:
                # Activity 1 is more critical, keep it closer to center, move activity 2
                adjusted_pos = self.adjust_activity_position(u2, v2, adjusted_pos, offset=min_arrow_separation)
            else:
                # Activity 2 is more critical, move activity 1
                adjusted_pos = self.adjust_activity_position(u1, v1, adjusted_pos, offset=min_arrow_separation)

        return adjusted_pos

    def are_arrows_overlapping(self, u1, v1, u2, v2, pos, min_separation):
        """
        Check if two arrows (u1→v1 and u2→v2) are visually overlapping.
        """
        # Get positions
        x1_start, y1_start = pos[u1]
        x1_end, y1_end = pos[v1]
        x2_start, y2_start = pos[u2]
        x2_end, y2_end = pos[v2]

        # Check if arrows are in similar X range (overlapping horizontal spans)
        x1_min, x1_max = min(x1_start, x1_end), max(x1_start, x1_end)
        x2_min, x2_max = min(x2_start, x2_end), max(x2_start, x2_end)

        # No horizontal overlap means no visual overlap
        if x1_max < x2_min or x2_max < x1_min:
            return False

        # Check if paths are too close vertically
        # Simple approach: check if start and end points are close
        start_distance = abs(y1_start - y2_start)
        end_distance = abs(y1_end - y2_end)

        # Arrows overlap if both start and end are too close
        return start_distance < min_separation and end_distance < min_separation

    def adjust_activity_position(self, start_node, end_node, pos, offset):
        """
        Adjust positions of start and/or end nodes to separate an activity arrow.
        """
        adjusted_pos = pos.copy()

        # Strategy: Move the end node (less disruptive than moving start node)
        # Choose direction based on current position relative to center
        current_y = pos[end_node][1]

        if current_y >= 0:
            # Node is above or at center, move it further up
            new_y = current_y + offset
        else:
            # Node is below center, move it further down
            new_y = current_y - offset

        adjusted_pos[end_node] = (pos[end_node][0], new_y)

        return adjusted_pos

    def count_arrow_crossings(self, G, pos):
        """
        Count the number of arrow crossings in the current layout.
        Two arrows cross if their line segments intersect.
        """
        crossings = 0
        edges = list(G.edges())
        
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                u1, v1 = edges[i]
                u2, v2 = edges[j]
                
                # Get positions
                x1, y1 = pos[u1]
                x2, y2 = pos[v1]
                x3, y3 = pos[u2]
                x4, y4 = pos[v2]
                
                # Check if line segments (x1,y1)-(x2,y2) and (x3,y3)-(x4,y4) intersect
                if self.lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
                    crossings += 1
                    
        return crossings

    def lines_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Check if two line segments intersect using the orientation method.
        """
        def orientation(px, py, qx, qy, rx, ry):
            """Find orientation of ordered triplet (p, q, r)"""
            val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
            if val == 0:
                return 0  # collinear
            return 1 if val > 0 else 2  # clockwise or counterclockwise

        def on_segment(px, py, qx, qy, rx, ry):
            """Check if point q lies on segment pr"""
            return (qx <= max(px, rx) and qx >= min(px, rx) and
                    qy <= max(py, ry) and qy >= min(py, ry))

        o1 = orientation(x1, y1, x2, y2, x3, y3)
        o2 = orientation(x1, y1, x2, y2, x4, y4)
        o3 = orientation(x3, y3, x4, y4, x1, y1)
        o4 = orientation(x3, y3, x4, y4, x2, y2)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special cases for collinear points
        if (o1 == 0 and on_segment(x1, y1, x3, y3, x2, y2) or
            o2 == 0 and on_segment(x1, y1, x4, y4, x2, y2) or
            o3 == 0 and on_segment(x3, y3, x1, y1, x4, y4) or
            o4 == 0 and on_segment(x3, y3, x2, y2, x4, y4)):
            return True

        return False

    def minimize_arrow_crossings(self, G, pos, event_floats, max_iterations=5):
        """
        Minimize arrow crossings using multiple strategies:
        1. Critical path alignment optimization
        2. Barycenter heuristic
        3. Iterative improvement
        """
        print("Minimizing arrow crossings...")
        
        current_pos = pos.copy()
        initial_crossings = self.count_arrow_crossings(G, current_pos)
        print(f"  Initial crossings: {initial_crossings}")
        
        if initial_crossings == 0:
            print("  No crossings to minimize!")
            return current_pos
        
        best_pos = current_pos.copy()
        best_crossings = initial_crossings
        
        # Strategy 1: Strategic node alignment optimization
        alignment_pos = self.try_critical_path_alignment(G, current_pos, event_floats)
        alignment_crossings = self.count_arrow_crossings(G, alignment_pos)
        print(f"  After strategic alignment: {alignment_crossings} crossings")
        
        if alignment_crossings < best_crossings:
            best_pos = alignment_pos.copy()
            best_crossings = alignment_crossings
        
        # Strategy 2: Barycenter heuristic with iterative improvement
        barycenter_pos = self.apply_barycenter_heuristic(G, best_pos, max_iterations)
        barycenter_crossings = self.count_arrow_crossings(G, barycenter_pos)
        print(f"  After barycenter optimization: {barycenter_crossings} crossings")
        
        if barycenter_crossings < best_crossings:
            best_pos = barycenter_pos.copy()
            best_crossings = barycenter_crossings
        
        improvement = initial_crossings - best_crossings
        if improvement > 0:
            print(f"  Successfully reduced crossings by {improvement} ({improvement/initial_crossings*100:.1f}%)")
            return best_pos
        else:
            # Even if crossings didn't improve, keep consolidation improvements for better aesthetics
            print("  No crossing improvement, but keeping node consolidation for better layout")
            return alignment_pos  # Return the position after strategic alignment (includes consolidation)

    def try_critical_path_alignment(self, G, pos, event_floats):
        """
        Try various alignment strategies to reduce crossings.
        This addresses the specific suggestion about aligning E2, E5 with E0, E1.
        """
        print("    Trying strategic node alignment...")
        optimized_pos = pos.copy()
        best_crossings = self.count_arrow_crossings(G, optimized_pos)
        
        # Strategy 1: Align nodes with similar topological positions
        optimized_pos = self.try_topological_alignment(G, optimized_pos, best_crossings)
        
        # Strategy 2: Try horizontal line arrangements
        optimized_pos = self.try_horizontal_line_arrangements(G, optimized_pos)
        
        # Strategy 3: Move non-critical nodes closer to critical path when possible
        optimized_pos = self.consolidate_non_critical_nodes(G, optimized_pos, event_floats)
        
        return optimized_pos

    def try_topological_alignment(self, G, pos, initial_crossings):
        """
        Try aligning nodes that are topologically close to reduce crossings.
        """
        optimized_pos = pos.copy()
        
        # Group nodes by topological level (X-coordinate)  
        levels = {}
        for node, (x, y) in pos.items():
            if x not in levels:
                levels[x] = []
            levels[x].append(node)
        
        # For each level, try aligning nodes with nodes from adjacent levels
        level_keys = sorted(levels.keys())
        
        for i, level_x in enumerate(level_keys):
            current_level_nodes = levels[level_x]
            
            # Try aligning with previous level
            if i > 0:
                prev_level_nodes = levels[level_keys[i-1]]
                for node in current_level_nodes:
                    for prev_node in prev_level_nodes:
                        test_pos = optimized_pos.copy()
                        test_pos[node] = (pos[node][0], pos[prev_node][1])
                        
                        test_crossings = self.count_arrow_crossings(G, test_pos)
                        current_crossings = self.count_arrow_crossings(G, optimized_pos)
                        
                        if test_crossings < current_crossings:
                            optimized_pos = test_pos.copy()
                            print(f"    Aligned E{node} with E{prev_node}: {current_crossings} → {test_crossings} crossings")
        
        return optimized_pos

    def try_horizontal_line_arrangements(self, G, pos):
        """
        Try arranging nodes in horizontal lines to minimize crossings.
        This specifically addresses cases like aligning E2, E5 with E0, E1.
        Uses a layered approach similar to Sugiyama algorithm.
        """
        print("    Trying layered horizontal arrangements...")
        optimized_pos = pos.copy()
        
        # Group nodes by topological level (X-coordinate)
        levels = {}
        for node, (x, y) in pos.items():
            if x not in levels:
                levels[x] = []
            levels[x].append(node)
        
        level_keys = sorted(levels.keys())
        
        # Strategy 1: Try aligning with main flow (critical path)
        optimized_pos = self.try_main_flow_alignment(G, optimized_pos, levels, level_keys)
        
        # Strategy 2: Layer-by-layer crossing minimization
        optimized_pos = self.minimize_crossings_between_layers(G, optimized_pos, levels, level_keys)
        
        return optimized_pos

    def try_main_flow_alignment(self, G, pos, levels, level_keys):
        """
        Try aligning nodes with the main flow (typically Y=0 or critical path).
        This addresses the specific E2, E5 with E0, E1 alignment scenario.
        """
        optimized_pos = pos.copy()
        current_crossings = self.count_arrow_crossings(G, optimized_pos)
        
        # Find the main flow Y-coordinate (most common Y or Y=0)
        all_y_coords = [y for x, y in pos.values()]
        main_flow_y = 0.0  # Default to Y=0 (critical path)
        
        # Alternative: use the most common Y-coordinate
        # from collections import Counter
        # y_counter = Counter(all_y_coords)
        # main_flow_y = y_counter.most_common(1)[0][0]
        
        print(f"      Main flow alignment at Y={main_flow_y:.2f}")
        
        # For each level, try moving nodes to main flow
        for level_x in level_keys:
            nodes_at_level = levels[level_x]
            
            if len(nodes_at_level) <= 1:
                continue
                
            for node in nodes_at_level:
                if abs(pos[node][1] - main_flow_y) < 0.1:
                    continue  # Already at main flow
                
                # Test moving this node to main flow
                test_pos = optimized_pos.copy()
                test_pos[node] = (pos[node][0], main_flow_y)
                
                test_crossings = self.count_arrow_crossings(G, test_pos)
                
                if test_crossings < current_crossings:
                    optimized_pos = test_pos.copy()
                    current_crossings = test_crossings
                    
                    # Count nodes now aligned at main flow
                    aligned_count = sum(1 for n, (x, y) in optimized_pos.items() 
                                      if abs(y - main_flow_y) < 0.1)
                    print(f"      Aligned E{node} to main flow: {test_crossings} crossings, {aligned_count} nodes aligned")
        
        return optimized_pos

    def minimize_crossings_between_layers(self, G, pos, levels, level_keys):
        """
        Apply layer-by-layer crossing minimization using median/barycenter heuristic.
        """
        optimized_pos = pos.copy()
        
        # Perform several passes of crossing reduction
        for iteration in range(3):
            improved = False
            
            # Forward pass: adjust each level based on previous level
            for i in range(1, len(level_keys)):
                current_level = levels[level_keys[i]]
                prev_level = levels[level_keys[i-1]]
                
                if len(current_level) <= 1:
                    continue
                    
                # Calculate optimal positions using barycenter heuristic
                for node in current_level:
                    # Find predecessors in previous level
                    predecessors = [pred for pred in G.predecessors(node) if pred in prev_level]
                    
                    if predecessors:
                        # Calculate barycenter (average Y of predecessors)
                        pred_y_coords = [optimized_pos[pred][1] for pred in predecessors]
                        barycenter_y = sum(pred_y_coords) / len(pred_y_coords)
                        
                        # Test if moving to barycenter reduces crossings
                        test_pos = optimized_pos.copy()
                        test_pos[node] = (optimized_pos[node][0], barycenter_y)
                        
                        current_crossings = self.count_arrow_crossings(G, optimized_pos)
                        test_crossings = self.count_arrow_crossings(G, test_pos)
                        
                        if test_crossings < current_crossings:
                            optimized_pos = test_pos.copy()
                            improved = True
                            print(f"      Layer {i}: moved E{node} to barycenter Y={barycenter_y:.2f}")
            
            if not improved:
                break
        
        return optimized_pos

    def consolidate_non_critical_nodes(self, G, pos, event_floats):
        """
        Move non-critical (green) nodes closer to the critical path when there are
        no competing higher-priority nodes in the region. This addresses cases like
        E3 being far from other nodes when it could be closer to the main flow.
        Uses incremental movement to avoid overly conservative blocking.
        """
        print("    Consolidating non-critical nodes toward critical path...")
        optimized_pos = pos.copy()
        
        # Find the critical path Y-coordinate (average of critical nodes)
        critical_nodes = [node for node, float_val in event_floats.items() if float_val == 0]
        if not critical_nodes:
            return optimized_pos
            
        critical_y_coords = [pos[node][1] for node in critical_nodes]
        critical_path_y = sum(critical_y_coords) / len(critical_y_coords)
        
        print(f"      Critical path at Y={critical_path_y:.2f}")
        
        # Categorize all nodes by criticality
        node_categories = {}
        for node in G.nodes():
            float_val = event_floats[node]
            category, color, is_bold = self.get_criticality_category(float_val)
            node_categories[node] = category
        
        # Find non-critical nodes that are far from critical path (include orange nodes too)
        non_critical_nodes = [(node, pos[node]) for node, category in node_categories.items() 
                             if category in ['uncritical', 'medium_critical']]
        
        for node, (x, y) in non_critical_nodes:
            distance_from_critical = abs(y - critical_path_y)
            
            # Only consider nodes that are significantly far from critical path
            if distance_from_critical < 1.0:
                continue
                
            print(f"      Checking E{node} ({node_categories[node]}): currently at Y={y:.2f}, distance={distance_from_critical:.2f}")
            
            # Use incremental consolidation instead of all-or-nothing approach
            best_position = self.find_best_consolidation_position(
                node, y, critical_path_y, pos, node_categories, G, optimized_pos
            )
            
            if best_position != y:
                # Test if this positioning improves or maintains crossing count
                test_pos = optimized_pos.copy()
                test_pos[node] = (x, best_position)
                
                original_crossings = self.count_arrow_crossings(G, optimized_pos)
                test_crossings = self.count_arrow_crossings(G, test_pos)
                
                if test_crossings <= original_crossings:  # Allow same number of crossings
                    optimized_pos = test_pos.copy()
                    new_distance = abs(best_position - critical_path_y)
                    improvement = distance_from_critical - new_distance
                    print(f"      ✓ Moved E{node} from Y={y:.2f} to Y={best_position:.2f} (improvement: {improvement:.2f})")
                else:
                    print(f"      ✗ Moving E{node} would increase crossings ({original_crossings} → {test_crossings}), skipping")
            else:
                print(f"      ✗ No safe consolidation position found for E{node}")
        
        return optimized_pos

    def find_best_consolidation_position(self, node, current_y, target_y, pos, node_categories, G, current_layout):
        """
        Find the best position to move a non-critical node toward the critical path
        using incremental steps instead of all-or-nothing blocking logic.
        """
        min_separation = 1.0  # Minimum distance from higher-priority nodes
        step_size = 0.5  # Incremental step size toward critical path
        
        # Determine direction toward critical path
        direction = 1 if target_y > current_y else -1
        
        # Try incremental positions toward critical path
        best_y = current_y
        node_x = pos[node][0]
        
        # Start with small steps and gradually move toward target
        test_y = current_y
        while abs(test_y - target_y) > step_size:
            # Take a step toward critical path
            test_y += direction * step_size
            
            # Check if this position has conflicts with higher-priority nodes
            safe_position = self.is_position_safe_from_conflicts(
                node, test_y, pos, node_categories, min_separation
            )
            
            if safe_position:
                best_y = test_y
            else:
                # Hit a conflict, stop here
                break
        
        # If we can get very close to target, try the exact target
        if abs(best_y - target_y) <= step_size:
            if self.is_position_safe_from_conflicts(node, target_y, pos, node_categories, min_separation):
                best_y = target_y
        
        return best_y

    def is_position_safe_from_conflicts(self, node, test_y, pos, node_categories, min_separation):
        """
        Check if a specific Y position is safe from conflicts with higher-priority nodes.
        Much less conservative than checking entire regions.
        """
        node_x = pos[node][0]
        
        # Check all other nodes for conflicts
        for other_node, (x, y) in pos.items():
            if other_node == node:
                continue
            
            # Only check nodes in same or adjacent topological levels
            if abs(x - node_x) <= 3.0:
                other_category = node_categories[other_node]
                
                # Check conflict with higher-priority nodes
                if other_category in ['critical', 'near_critical']:
                    if abs(test_y - y) < min_separation:
                        return False
                
                # Be more lenient with medium_critical nodes
                elif other_category == 'medium_critical':
                    if abs(test_y - y) < min_separation * 0.7:  # Allow closer approach
                        return False
        
        return True

    def is_region_clear_of_higher_priority_nodes(self, node, current_y, target_y, pos, node_categories):
        """
        Check if the region between current_y and target_y is clear of nodes with
        higher priority (critical, near_critical, medium_critical).
        """
        min_y, max_y = min(current_y, target_y), max(current_y, target_y)
        
        # Check all other nodes in this topological level and adjacent levels
        node_x = pos[node][0]
        
        for other_node, (x, y) in pos.items():
            if other_node == node:
                continue
                
            # Check nodes in same level or adjacent levels (within 3 units on X-axis)
            if abs(x - node_x) <= 3.0:
                # Check if this node is in the consolidation region
                if min_y <= y <= max_y:
                    other_category = node_categories[other_node]
                    # Higher priority nodes: critical, near_critical, medium_critical
                    if other_category in ['critical', 'near_critical', 'medium_critical']:
                        print(f"        Region blocked by E{other_node} ({other_category}) at Y={y:.2f}")
                        return False
        
        return True

    def calculate_safe_consolidation_position(self, node, current_y, critical_path_y, pos, G):
        """
        Calculate a safe Y position for consolidating a non-critical node toward
        the critical path, considering collision avoidance.
        """
        # Start by moving halfway toward critical path
        target_y = (current_y + critical_path_y) / 2
        
        # Check for potential collisions and adjust if needed
        node_x = pos[node][0]
        min_separation = 0.8
        
        # Find other nodes at the same X level
        same_level_nodes = [(other_node, y) for other_node, (x, y) in pos.items() 
                           if other_node != node and abs(x - node_x) < 0.1]
        
        if same_level_nodes:
            same_level_y_coords = [y for _, y in same_level_nodes]
            
            # Find a safe Y position that maintains minimum separation
            for test_y in [target_y, critical_path_y + 0.5, critical_path_y - 0.5]:
                safe = True
                for other_y in same_level_y_coords:
                    if abs(test_y - other_y) < min_separation:
                        safe = False
                        break
                if safe:
                    target_y = test_y
                    break
        
        return target_y

    def apply_barycenter_heuristic(self, G, pos, max_iterations=3):
        """
        Apply barycenter heuristic: position each node at the average Y-coordinate of its neighbors.
        This tends to reduce crossings by creating more "natural" positioning.
        """
        print("    Applying barycenter heuristic...")
        current_pos = pos.copy()
        
        for iteration in range(max_iterations):
            improved = False
            new_pos = current_pos.copy()
            
            # Group nodes by topological level (X-coordinate)
            levels = {}
            for node, (x, y) in current_pos.items():
                if x not in levels:
                    levels[x] = []
                levels[x].append(node)
            
            # Process each level
            for level_x, nodes_at_level in levels.items():
                if len(nodes_at_level) <= 1:
                    continue
                
                for node in nodes_at_level:
                    # Calculate barycenter (average Y of neighbors)
                    neighbor_y_coords = []
                    
                    # Add predecessor Y-coordinates
                    for pred in G.predecessors(node):
                        neighbor_y_coords.append(current_pos[pred][1])
                    
                    # Add successor Y-coordinates
                    for succ in G.successors(node):
                        neighbor_y_coords.append(current_pos[succ][1])
                    
                    if neighbor_y_coords:
                        barycenter_y = sum(neighbor_y_coords) / len(neighbor_y_coords)
                        
                        # Test if moving to barycenter reduces crossings
                        test_pos = current_pos.copy()
                        test_pos[node] = (current_pos[node][0], barycenter_y)
                        
                        current_crossings = self.count_arrow_crossings(G, current_pos)
                        test_crossings = self.count_arrow_crossings(G, test_pos)
                        
                        if test_crossings < current_crossings:
                            new_pos[node] = (current_pos[node][0], barycenter_y)
                            improved = True
            
            current_pos = new_pos
            
            if not improved:
                print(f"    Converged after {iteration + 1} iterations")
                break
        
        return current_pos

    def draw_aoa_network_float_based(self, title="AoA Network (Float-Based Layout)", criticality_thresholds=None):
        """
        Draw AoA network using float-based vertical positioning
        """
        fig, ax = plt.subplots(figsize=(20, 12))
        G = self.golenko_ginzburg_direct(self.tasks)

        # Create float-based layout
        pos, event_floats = self.create_float_based_layout(G)

        print(f"Layout: Float-based vertical positioning")

        # Separate real and dummy activities
        real_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('is_dummy', False)]
        dummy_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_dummy', False)]

        # Get task floats for activity classification
        task_floats = self.get_task_floats()

        # Draw events (nodes) with unfilled circles and colored borders
        node_edge_colors = []
        node_linewidths = []

        for node in G.nodes():
            float_val = event_floats[node]
            category, color, is_bold = self.get_criticality_category(float_val, criticality_thresholds)
            node_edge_colors.append(color)
            node_linewidths.append(3 if is_bold else 2)

        node_collection = nx.draw_networkx_nodes(G, pos,
                            node_color='white',  # Unfilled nodes
                            node_size=1000,
                            edgecolors=node_edge_colors,  # Colored borders
                            linewidths=node_linewidths,
                            ax=ax)

        # Store mapping from node ID to (artist, (x, y) position)
        self.aoa_node_artists = {}
        for idx, node in enumerate(G.nodes()):
            # node_collection is a PathCollection; get_offsets() gives all positions
            xy = pos[node]
            # Each node is a PathCollection, but we can use the collection and index
            self.aoa_node_artists[node] = (node_collection, xy)

        # Store the node order for interactive scripts
        self.aoa_node_order = list(G.nodes())

        # Event labels with criticality-based text coloring
        for node in G.nodes():
            float_val = event_floats[node]
            category, color, is_bold = self.get_criticality_category(float_val, criticality_thresholds)
            label = f"E{node}"
            x, y = pos[node]

            ax.text(x, y, label,
                ha='center', va='center',
                color=color,  # Colored text
                fontsize=10,
                fontweight='bold' if is_bold else 'normal',
                zorder=5)

        # Draw activity arrows with criticality-based styling
        if real_edges:
            # Group activities by criticality category
            activity_groups = {
                'critical': [],
                'near_critical': [],
                'medium_critical': [],
                'uncritical': []
            }

            for u, v in real_edges:
                edge_data = G[u][v]
                activity_id = edge_data.get('activity_id')
                float_val = task_floats.get(activity_id, 0)
                category, color, is_bold = self.get_criticality_category(float_val, criticality_thresholds)
                activity_groups[category].append((u, v))

            # Store edge artists for interactivity
            self.aoa_edge_artists = {}
            for category, edges in activity_groups.items():
                if not edges:
                    continue

                if category == 'critical':
                    color, width, arrowsize = 'black', 4, 25  # Bold black
                elif category == 'near_critical':
                    color, width, arrowsize = 'red', 2, 20    # Non-bold red
                elif category == 'medium_critical':
                    color, width, arrowsize = 'orange', 2, 20 # Non-bold orange
                else:  # uncritical
                    color, width, arrowsize = 'green', 2, 20  # Non-bold green

                # Separate edges by duration (0 duration = dummy activity = dashed line)
                solid_edges = []
                dashed_edges = []

                for u, v in edges:
                    edge_data = G[u][v]
                    duration = edge_data.get('duration', 0)
                    if duration == 0:
                        dashed_edges.append((u, v))
                    else:
                        solid_edges.append((u, v))

                # Draw solid edges (normal activities)
                if solid_edges:
                    edge_artists = nx.draw_networkx_edges(G, pos, edgelist=solid_edges,
                                        edge_color=color,
                                        arrows=True,
                                        arrowsize=arrowsize,
                                        arrowstyle='->',
                                        width=width,
                                        style='solid',
                                        ax=ax)
                    for i, (u, v) in enumerate(solid_edges):
                        self.aoa_edge_artists[(u, v)] = edge_artists[i]

                # Draw dashed edges (0 duration activities)
                if dashed_edges:
                    edge_artists = nx.draw_networkx_edges(G, pos, edgelist=dashed_edges,
                                        edge_color=color,
                                        arrows=True,
                                        arrowsize=arrowsize,
                                        arrowstyle='->',
                                        width=width,
                                        style='dashed',
                                        ax=ax)
                    for i, (u, v) in enumerate(dashed_edges):
                        self.aoa_edge_artists[(u, v)] = edge_artists[i]

        # Draw dummy arrows (dashed)
        if dummy_edges:
            edge_artists = nx.draw_networkx_edges(G, pos, edgelist=dummy_edges,
                                edge_color='gray',
                                arrows=True,
                                arrowsize=arrowsize,
                                arrowstyle='->',
                                width=width,
                                style='dashed',
                                ax=ax)
            for i, (u, v) in enumerate(dummy_edges):
                self.aoa_edge_artists[(u, v)] = edge_artists[i]

        # Activity labels with criticality-based coloring
        self.aoa_label_artists = {}
        for u, v, data in G.edges(data=True):
            if not data.get('is_dummy', False):
                name = data['activity_name']
                duration = data['duration']
                activity_id = data.get('activity_id')
                float_val = task_floats.get(activity_id, 0)

                # Get criticality styling
                category, color, is_bold = self.get_criticality_category(float_val, criticality_thresholds)

                # Create label text
                short_name = name if len(name) <= 8 else name[:6] + ".."
                label_text = f"{short_name}\n({duration}d, {float_val}f)"

                # Calculate label position (midpoint of edge)
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                label_x = (x1 + x2) / 2
                label_y = (y1 + y2) / 2

                # Draw label with criticality-based coloring
                label_artist = ax.text(label_x, label_y, label_text,
                    ha='center', va='center',
                    color=color,  # Criticality-based color
                    fontsize=8,
                    fontweight='bold' if is_bold else 'normal',
                    bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor=color,  # Colored border
                            alpha=0.9,
                            linewidth=2 if is_bold else 1),
                    zorder=6)
                self.aoa_label_artists[(u, v)] = label_artist

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        # Enhanced legend for criticality-based layout
        if criticality_thresholds is None:
            thresholds = {'near_critical': 5, 'medium_critical': 15}
        else:
            thresholds = criticality_thresholds

        legend_text = (
            "Criticality-Based AoA Layout:\n"
            f"• BLACK (Bold): Critical path (0 float)\n"
            f"• RED: Near critical (1-{thresholds['near_critical']} float)\n"
            f"• ORANGE: Medium critical (6-{thresholds['medium_critical']} float)\n"
            f"• GREEN: Uncritical (>{thresholds['medium_critical']} float)\n"
            "• Bold = Critical path emphasis\n"
            "• Dashed lines = 0 duration activities\n"
            "• Vertical position ∝ Float value\n"
            "• Critical path centered vertically\n"
            "• E0 = Project start"
        )

        plt.text(1.02, 0.5, legend_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        plt.tight_layout()
        # plt.show()  # Remove this to allow caller to set up interactivity first

        return G

    def enable_aoa_network_interactivity(self, *args, **kwargs):
        """
        Draw the AoA network and enable interactive node dragging using matplotlib's event API.
        """
        # Draw the network and populate self.aoa_node_artists
        G = self.draw_aoa_network_float_based(*args, **kwargs)
        
        # Get the current figure and axis
        fig = plt.gcf()
        ax = plt.gca()
        
        # State for dragging
        self._dragging_node = None
        self._drag_offset = (0, 0)
        self._drag_artist = None
        self._drag_original_pos = None
        self._aoa_interactive_enabled = True
        self._drag_highlight = None
        
        # Helper: hit test for node
        def hit_test_node(event):
            if event.inaxes != ax:
                return None
            # Use a tolerance in data coordinates
            tolerance = 0.5  # Adjust as needed
            for node, (artist, (x, y)) in self.aoa_node_artists.items():
                dx = event.xdata - x
                dy = event.ydata - y
                if (dx * dx + dy * dy) ** 0.5 < tolerance:
                    return node
            return None
        
        def emphasize_node(node, emphasize=True):
            artist, (x, y) = self.aoa_node_artists[node]
            offsets = artist.get_offsets()
            node_list = list(self.aoa_node_artists.keys())
            idx = node_list.index(node)
            orig_size = artist.get_sizes()[idx]
            orig_color = artist.get_facecolors()[idx] if len(artist.get_facecolors()) > idx else [1,1,1,1]
            if emphasize:
                # Increase size and change color
                sizes = artist.get_sizes()
                sizes[idx] = orig_size * 1.5
                artist.set_sizes(sizes)
                # Set facecolor to yellow for highlight
                facecolors = artist.get_facecolors()
                if len(facecolors) <= idx:
                    # Expand facecolors if needed
                    import numpy as np
                    new_fc = np.tile([1,1,1,1], (len(offsets),1))
                    facecolors = new_fc
                facecolors[idx] = [1, 1, 0, 1]  # yellow
                artist.set_facecolors(facecolors)
                # Bring to front
                artist.set_zorder(10)
            else:
                # Restore size and color
                sizes = artist.get_sizes()
                sizes[idx] = 1000
                artist.set_sizes(sizes)
                facecolors = artist.get_facecolors()
                facecolors[idx] = [1, 1, 1, 1]  # white
                artist.set_facecolors(facecolors)
                artist.set_zorder(1)
            fig.canvas.draw_idle()
        
        def on_press(event):
            if not self._aoa_interactive_enabled:
                return
            node = hit_test_node(event)
            if node is not None:
                self._dragging_node = node
                self._drag_artist = self.aoa_node_artists[node][0]
                self._drag_original_pos = self.aoa_node_artists[node][1]
                x, y = self._drag_original_pos
                self._drag_offset = (event.xdata - x, event.ydata - y)
                emphasize_node(node, True)
        
        def on_motion(event):
            if not self._aoa_interactive_enabled:
                return
            if self._dragging_node is not None and event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                node = self._dragging_node
                artist, _ = self.aoa_node_artists[node]
                # Update position
                new_x = event.xdata - self._drag_offset[0]
                new_y = event.ydata - self._drag_offset[1]
                # Update the stored position
                self.aoa_node_artists[node] = (artist, (new_x, new_y))
                # Update the PathCollection offsets
                offsets = artist.get_offsets()
                node_list = list(self.aoa_node_artists.keys())
                idx = node_list.index(node)
                offsets[idx] = [new_x, new_y]
                artist.set_offsets(offsets)

                # Update all connected edges and labels
                for (u, v), edge_artist in self.aoa_edge_artists.items():
                    if u == node or v == node:
                        # Update the edge line data
                        x1, y1 = self.aoa_node_artists[u][1]
                        x2, y2 = self.aoa_node_artists[v][1]
                        edge_artist.set_data([x1, x2], [y1, y2])
                for (u, v), label_artist in self.aoa_label_artists.items():
                    if u == node or v == node:
                        x1, y1 = self.aoa_node_artists[u][1]
                        x2, y2 = self.aoa_node_artists[v][1]
                        label_x = (x1 + x2) / 2
                        label_y = (y1 + y2) / 2
                        label_artist.set_position((label_x, label_y))

                fig.canvas.draw_idle()
        
        def on_release(event):
            if not self._aoa_interactive_enabled:
                return
            if self._dragging_node is not None:
                emphasize_node(self._dragging_node, False)
                self._dragging_node = None
                self._drag_artist = None
                self._drag_original_pos = None
                self._drag_offset = (0, 0)
        
        # Save Layout button
        def save_layout(event=None):
            layout = {int(node): {'x': float(pos[0]), 'y': float(pos[1])} for node, (_, pos) in self.aoa_node_artists.items()}
            with open('aoa_layout.json', 'w') as f:
                json.dump(layout, f, indent=2)
            print("Layout saved to aoa_layout.json")
        
        # Add a simple button
        button_ax = fig.add_axes([0.85, 0.01, 0.12, 0.05])
        save_button = Button(button_ax, 'Save Layout')
        save_button.on_clicked(save_layout)
        
        # Optionally, add a keyboard shortcut (e.g., press 's' to save)
        def on_key(event):
            if event.key == 's':
                save_layout()
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Connect the events
        self._aoa_cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
        self._aoa_cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
        self._aoa_cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
        
        # Optionally, return the figure and axis for further use
        return fig, ax

    def topological_sort_tasks(self, tasks):
        """Sort tasks in topological order using their predecessor relationships."""
        task_dict = {task["ID"]: task for task in tasks}
        in_degree = {task["ID"]: len(task["Predecessors"]) for task in tasks}

        # Start with tasks that have no predecessors
        queue = [task for task in tasks if not task["Predecessors"]]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Find all tasks that have current as a predecessor
            for task in tasks:
                if current["ID"] in task["Predecessors"]:
                    in_degree[task["ID"]] -= 1
                    if in_degree[task["ID"]] == 0:
                        queue.append(task)

        return result

    def find_task_successors(self, task_id, tasks):
        """Find all immediate successors of a given task."""
        return [task["ID"] for task in tasks if task_id in task["Predecessors"]]

    def determine_start_event_direct(self, predecessors, activity_events, aoa_network, event_counter):
        """Determine start event using direct GG rules."""
        if not predecessors:
            return 0  # Project start

        # Get end events of all predecessors
        pred_end_events = set()
        for pred_id in predecessors:
            if pred_id in activity_events:
                _, pred_end = activity_events[pred_id]
                pred_end_events.add(pred_end)

        # If all predecessors end at same event, use it
        if len(pred_end_events) == 1:
            return list(pred_end_events)[0]

        # Multiple predecessor end events - check for existing convergence
        for existing_event in aoa_network.nodes():
            incoming = set(aoa_network.predecessors(existing_event))
            if pred_end_events.issubset(incoming):
                return existing_event

        # Need new convergence event
        return event_counter

    def determine_end_event_direct(self, task_id, successors, activity_events, event_end_activities, tasks, event_counter):
        """Determine end event by analyzing successor patterns."""
        successor_set = set(successors)

        # Look for existing events with same successor pattern
        for event, activities in event_end_activities.items():
            if activities:  # Check existing activities at this event
                existing_activity = next(iter(activities))
                existing_successors = set(self.find_task_successors(existing_activity, tasks))
                if existing_successors == successor_set:
                    return event

        # No matching pattern - create new event
        return event_counter

    def add_dummy_activities_direct(self, tasks, activity_events, aoa_network):
        """Add minimal dummy activities to maintain precedence."""
        dummy_count = 0

        for task in tasks:
            task_id = task["ID"]
            _, task_end = activity_events[task_id]

            # Check all successors
            for successor_task in tasks:
                if task_id in successor_task["Predecessors"]:
                    successor_start, _ = activity_events[successor_task["ID"]]

                    # Need dummy if events differ and no arrow exists
                    if task_end != successor_start:
                        if not aoa_network.has_edge(task_end, successor_start):
                            aoa_network.add_edge(task_end, successor_start,
                                               activity_id=f"dummy_{task_id}_to_{successor_task['ID']}",
                                               activity_name="(dummy)",
                                               duration=0,
                                               is_dummy=True)
                            dummy_count += 1
                            print(f"  Dummy: E{task_end} → E{successor_start} (for {task_id} → {successor_task['ID']})")

        return dummy_count

    def golenko_ginzburg_direct(self, tasks):
        """
        Apply Golenko-Ginzburg algorithm directly to raw task dependency data.

        This is the original approach - no intermediate AoN network needed.
        Works directly with:
        - Task IDs, names, durations
        - Predecessor relationships

        Constructs minimal AoA network using GG principles:
        1. Events represent project milestones where multiple activities converge/diverge
        2. Activities share events when their dependency patterns permit
        3. Dummy activities created only when absolutely necessary for precedence
        """
        print("Direct Golenko-Ginzburg Algorithm (Working with Raw Dependencies)")
        print("=" * 65)

        # Create the AoA network
        aoa_network = nx.DiGraph()

        # Build task index for quick lookup
        task_dict = {task["ID"]: task for task in tasks}

        # Step 1: Topological ordering of tasks
        print("Step 1: Determining task processing order...")
        ordered_tasks = self.topological_sort_tasks(tasks)
        print(f"Processing order: {[t['ID'] for t in ordered_tasks]}")

        # Step 2: Initialize event tracking
        event_counter = 0
        project_start = event_counter
        event_counter += 1

        activity_events = {}  # Maps task_id → (start_event, end_event)
        event_end_activities = defaultdict(set)  # Maps event → set of activities ending there

        print(f"\nProject start event: E{project_start}")

        # Step 3: Process each task in topological order
        print("\nStep 2: Assigning events using Golenko-Ginzburg rules...")

        for task in ordered_tasks:
            task_id = task["ID"]
            predecessors = task["Predecessors"]

            print(f"\nProcessing Task {task_id} ({task['Name']}):")
            print(f"  Predecessors: {predecessors}")

            # Determine start event
            if not predecessors:
                # No predecessors - starts at project start
                start_event = project_start
                print(f"  Start event: E{start_event} (project start)")
            else:
                # Apply GG rule for start event
                start_event = self.determine_start_event_direct(
                    predecessors, activity_events, aoa_network, event_counter
                )
                if start_event >= event_counter:
                    event_counter = start_event + 1
                print(f"  Start event: E{start_event}")

            # Determine end event using successor analysis
            successors = self.find_task_successors(task_id, tasks)
            end_event = self.determine_end_event_direct(
                task_id, successors, activity_events, event_end_activities,
                tasks, event_counter
            )
            if end_event >= event_counter:
                event_counter = end_event + 1

            print(f"  End event: E{end_event}")
            print(f"  Successors: {successors}")

            # Record the task's events
            activity_events[task_id] = (start_event, end_event)
            event_end_activities[end_event].add(task_id)

        # Step 4: Add task arrows to network
        print("\nStep 3: Adding activity arrows...")
        for task in tasks:
            task_id = task["ID"]
            start_evt, end_evt = activity_events[task_id]

            aoa_network.add_edge(start_evt, end_evt,
                               activity_id=task_id,
                               activity_name=f"{task_id} {task["Name"]}",
                               duration=task["Duration"],
                               is_dummy=False)

            print(f"  Task {task_id}: E{start_evt} → E{end_evt} ({task['Duration']}d)")

        # Step 5: Add minimal dummy activities for precedence
        print("\nStep 4: Adding necessary dummy activities...")
        dummy_count = self.add_dummy_activities_direct(tasks, activity_events, aoa_network)

        # Step 6: Handle project end
        terminal_tasks = [t for t in tasks if not self.find_task_successors(t["ID"], tasks)]
        terminal_events = set(activity_events[t["ID"]][1] for t in terminal_tasks)

        if len(terminal_events) > 1:
            project_end = event_counter
            print(f"\nProject end event: E{project_end}")
            for task in terminal_tasks:
                _, task_end = activity_events[task["ID"]]
                if task_end != project_end:
                    aoa_network.add_edge(task_end, project_end,
                                       activity_id=f"dummy_to_end_{task['ID']}",
                                       activity_name="(dummy)",
                                       duration=0,
                                       is_dummy=True)
                    dummy_count += 1
                    print(f"  Dummy to end: E{task_end} → E{project_end}")

        print(f"\nTotal dummy activities created: {dummy_count}")
        return aoa_network

    def tasks_overlap(self, task1: dict, task2: dict) -> bool:
        """
        Check if two tasks overlap in time.
        
        Args:
            task1: Dictionary representing the first task (must have 'Start' and 'Finish' keys)
            task2: Dictionary representing the second task (must have 'Start' and 'Finish' keys)
        
        Returns:
            True if the tasks overlap in time, False otherwise
        
        Raises:
            KeyError: If either task is missing 'Start' or 'Finish' keys
            TypeError: If the date values are not comparable
        """
        # Validate that both tasks have the required date fields
        required_fields = ['Start', 'Finish']
        for task, task_name in [(task1, 'task1'), (task2, 'task2')]:
            for field in required_fields:
                if field not in task:
                    raise KeyError(f"{task_name} is missing required field '{field}'")
        
        # Extract start and finish dates for both tasks
        task1_start = task1['Start']
        task1_finish = task1['Finish']
        task2_start = task2['Start']
        task2_finish = task2['Finish']
        
        # Tasks overlap if:
        # - Task1 starts before or when Task2 finishes AND
        # - Task2 starts before or when Task1 finishes
        # This is equivalent to: NOT (task1_finish < task2_start OR task2_finish < task1_start)
        return not (task1_finish < task2_start or task2_finish < task1_start)

    def is_person_available(self, person_name: str, new_task: dict, person_assignments: dict, tasks: list = None) -> bool:
        """
        Check if a person is available to take on a new task without conflicts.
        
        Args:
            person_name: Name of the person to check availability for
            new_task: Dictionary representing the new task (must have 'Start' and 'Finish' keys)
            person_assignments: Dictionary mapping person names to lists of assigned tasks
            tasks: Optional list of all tasks (unused in current implementation)
        
        Returns:
            True if the person is available (no overlapping tasks), False otherwise
        
        Raises:
            KeyError: If the new_task is missing 'Start' or 'Finish' keys
            TypeError: If the date values are not comparable
        """
        # Validate that new_task has the required date fields
        required_fields = ['Start', 'Finish']
        for field in required_fields:
            if field not in new_task:
                raise KeyError(f"new_task is missing required field '{field}'")
        
        # If person has no current assignments, they're available
        if person_name not in person_assignments or not person_assignments[person_name]:
            return True
        
        # Check for overlaps with currently assigned tasks
        assigned_tasks = person_assignments[person_name]
        for assigned_task in assigned_tasks:
            # Validate that assigned task has required fields
            for field in required_fields:
                if field not in assigned_task:
                    raise KeyError(f"assigned_task is missing required field '{field}'")
            
            # Use the existing tasks_overlap method to check for conflicts
            if self.tasks_overlap(new_task, assigned_task):
                return False  # Found an overlap, person is not available
        
        return True  # No overlaps found, person is available

    def assign_tasks_to_persons(self, tasks: list, task_resource_mapping: dict, resource_name_stems: dict, max_gap_days: int = 5) -> dict:
        """
        Assign tasks to persons based on resource requirements and availability.
        Implements the constraint that once a person leaves the project (due to work gaps), 
        they cannot rejoin to avoid on-and-off scenarios.
        
        Args:
            tasks: List of scheduled task dictionaries (must have 'Name', 'Start', 'Finish', 'TF' keys)
            task_resource_mapping: Dictionary mapping task names to required resource types
            resource_name_stems: Dictionary mapping resource types to name prefixes for person creation
            max_gap_days: Maximum gap in days before a person is considered to have left the project
        
        Returns:
            Dictionary mapping person names to lists of assigned tasks
        
        Raises:
            KeyError: If tasks are missing required fields or mappings are incomplete
        """
        # Step 1: Initialize empty dictionaries
        person_assignments = {}  # {person_name: [list_of_assigned_tasks]}
        existing_persons = {}    # {resource_type: [list_of_person_names]}
        person_last_finish = {}  # {person_name: last_finish_date} - track when person last worked
        people_who_left = set()  # Set of people who have left the project
        
        # Step 2: Get sorted task list (critical path first, then by start date, then by float)
        # This ensures we process tasks chronologically while prioritizing critical path
        sorted_tasks = sorted(tasks, key=lambda t: (
            t.get('TF', float('inf')),  # Critical path first (TF = 0)
            t.get('Start'),             # Then by start date
            t.get('Name', '')           # Then by name for consistency
        ))
        
        # Step 3: Process each task in priority order
        for task in sorted_tasks:
            task_name = task['Name']
            task_start = task['Start']
            
            # Validate task has required fields
            required_fields = ['Start', 'Finish', 'Name']
            for field in required_fields:
                if field not in task:
                    raise KeyError(f"Task '{task_name}' is missing required field '{field}'")
            
            # Get required resource type for this task
            if task_name not in task_resource_mapping:
                raise KeyError(f"No resource mapping found for task '{task_name}'")
            
            resource_type = task_resource_mapping[task_name]
            is_critical = task.get('TF', float('inf')) == 0
            
            # Step 4: Check who has left the project due to work gaps
            newly_left = []
            for person_name in list(person_last_finish.keys()):
                if person_name not in people_who_left:
                    last_finish = person_last_finish[person_name]
                    gap_days = (task_start - last_finish).days
                    if gap_days > max_gap_days:
                        people_who_left.add(person_name)
                        newly_left.append((person_name, gap_days))
            
            # Step 5: Find an available person of the correct resource type
            assigned_person = None
            
            # Check existing persons of this resource type
            if resource_type in existing_persons:
                available_people = [p for p in existing_persons[resource_type] if p not in people_who_left]
                
                for person_name in available_people:
                    if self.is_person_available(person_name, task, person_assignments):
                        assigned_person = person_name
                        break
            
            # Step 6: Create new person if no one is available
            if assigned_person is None:
                # Generate unique person name
                if resource_type not in resource_name_stems:
                    raise KeyError(f"No name stem found for resource type '{resource_type}'")
                
                name_stem = resource_name_stems[resource_type]
                if resource_type not in existing_persons:
                    existing_persons[resource_type] = []
                
                person_number = len(existing_persons[resource_type]) + 1
                assigned_person = f"{name_stem}_{person_number}"
                
                # Add to tracking dictionaries
                existing_persons[resource_type].append(assigned_person)
                person_assignments[assigned_person] = []
            
            # Step 7: Assign the task to the person
            if assigned_person not in person_assignments:
                person_assignments[assigned_person] = []
            
            person_assignments[assigned_person].append(task)
            
            # Update when this person last worked
            person_last_finish[assigned_person] = task['Finish']
        
        total_people_created = sum(len(people) for people in existing_persons.values())
        active_people = len(person_assignments)
        people_left = len(people_who_left)
        total_assignments = sum(len(tasks) for tasks in person_assignments.values())
        
        return person_assignments

    def validate_assignments(self, person_assignments: dict, tasks: list) -> bool:
        """
        Validate task assignments to ensure correctness and completeness.
        
        Args:
            person_assignments: Dictionary mapping person names to lists of assigned tasks
            tasks: List of all tasks that should be assigned
        
        Returns:
            True if assignments are valid, False otherwise
        
        Validation checks:
        1. No person has overlapping task assignments
        2. All tasks are assigned exactly once
        3. Task integrity (proper date fields)
        """
        
        validation_errors = []
        warnings = []
        
        # Check 1: No person has overlapping assignments
        overlap_errors = 0
        
        for person_name, assigned_tasks in person_assignments.items():
            if len(assigned_tasks) <= 1:
                continue  # Single task can't overlap with itself
            
            # Check all pairs of tasks for this person
            person_overlaps = []
            for i, task1 in enumerate(assigned_tasks):
                for j, task2 in enumerate(assigned_tasks):
                    if i < j:  # Only check each pair once
                        try:
                            if self.tasks_overlap(task1, task2):
                                overlap_errors += 1
                                overlap_msg = f"Tasks '{task1['Name']}' and '{task2['Name']}' overlap"
                                person_overlaps.append(overlap_msg)
                                validation_errors.append(f"{person_name}: {overlap_msg}")
                        except KeyError as e:
                            error_msg = f"Task missing required fields: {e}"
                            validation_errors.append(f"{person_name}: {error_msg}")
            
            if person_overlaps:
                print(f"      ❌ {len(person_overlaps)} overlap(s) found:")
                for overlap in person_overlaps:
                    print(f"         • {overlap}")
        
        if overlap_errors != 0:
            print(f"   ❌ Found {overlap_errors} overlapping assignments")
        
        # Check 2: All tasks are assigned exactly once
        
        # Collect all assigned tasks
        assigned_task_names = set()
        assigned_task_counts = {}
        
        for person_name, assigned_tasks in person_assignments.items():
            for task in assigned_tasks:
                task_name = task.get('Name', 'UNNAMED_TASK')
                assigned_task_names.add(task_name)
                assigned_task_counts[task_name] = assigned_task_counts.get(task_name, 0) + 1
        
        # Check if all input tasks are assigned
        input_task_names = set(task.get('Name', 'UNNAMED_TASK') for task in tasks)
        
        missing_tasks = input_task_names - assigned_task_names
        extra_tasks = assigned_task_names - input_task_names
        
        if missing_tasks:
            print(f"   ❌ {len(missing_tasks)} tasks not assigned:")
            for task_name in sorted(missing_tasks):
                print(f"      • {task_name}")
                validation_errors.append(f"Task '{task_name}' not assigned")
        
        if extra_tasks:
            print(f"   ⚠️  {len(extra_tasks)} extra tasks assigned:")
            for task_name in sorted(extra_tasks):
                print(f"      • {task_name}")
                warnings.append(f"Extra task assigned: '{task_name}'")
        
        # Check for duplicate assignments
        duplicate_assignments = {name: count for name, count in assigned_task_counts.items() if count > 1}
        if duplicate_assignments:
            print(f"   ❌ {len(duplicate_assignments)} tasks assigned multiple times:")
            for task_name, count in duplicate_assignments.items():
                print(f"      • {task_name}: {count} times")
                validation_errors.append(f"Task '{task_name}' assigned {count} times")
        
        # Final validation result
        print(f"\n" + "=" * 60)
        print("Validation result:")
        print("=" * 60)
        
        is_valid = len(validation_errors) == 0
        
        if is_valid:
            print(f"✅ VALIDATION PASSED")
            print(f"   • No overlapping assignments")
            print(f"   • All tasks assigned exactly once")
            print(f"   • {len(person_assignments)} people with valid assignments")
        else:
            print(f"❌ VALIDATION FAILED")
            print(f"   • {len(validation_errors)} error(s) found")
        
        if validation_errors:
            print(f"\n🚨 ERRORS FOUND:")
            for i, error in enumerate(validation_errors, 1):
                print(f"   {i}. {error}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS:")
            for i, warning in enumerate(warnings, 1):
                print(f"   {i}. {warning}")
        
        return is_valid

    def calculate_planned_earned_value(self, tasks: list, person_assignments: dict) -> dict:
        """
        Calculate planned earned value for all tasks.
        
        Args:
            tasks: List of scheduled task dictionaries (must have 'Name', 'Duration', 'Finish' keys)
            person_assignments: Dictionary mapping person names to lists of assigned tasks
        
        Returns:
            Dictionary containing:
            - 'total_duration': Sum of all task durations
            - 'sorted_tasks': Tasks sorted by completion date with earned value calculations
            - 'task_owners': Mapping of task names to assigned persons
            - 'milestones': List of earned value milestone information
            - 'summary_stats': Dictionary with project statistics
        
        Raises:
            ValueError: If tasks list is empty or contains invalid data
        """
        if not tasks:
            raise ValueError("Tasks list cannot be empty")
        
        # Step 1: Calculate the sum of all durations
        total_duration = sum(task.get('Duration', 0) for task in tasks)
        if total_duration <= 0:
            raise ValueError("Total duration must be greater than zero")
        
        # Step 2: Sort tasks by planned completion date (Finish date)
        sorted_tasks = sorted(tasks, key=lambda t: t.get('Finish', 'Z'))  # 'Z' as fallback for missing dates
        
        # Step 3: Find task owners from person assignments (reverse lookup)
        task_owners = {}
        for person, assigned_tasks in person_assignments.items():
            for task in assigned_tasks:
                task_owners[task['Name']] = person
        
        # Step 4: Calculate planned earned value for each task
        cumulative_duration = 0
        enhanced_tasks = []
        
        for i, task in enumerate(sorted_tasks):
            task_name = task.get('Name', f'Task_{i+1}')
            task_id = f"T{i+1:03d}"  # Generate task ID like T001, T002, etc.
            duration = task.get('Duration', 0)
            completion_date = task.get('Finish', 'N/A')
            owner = task_owners.get(task_name, 'Unassigned')
            
            # Add current task duration to cumulative
            cumulative_duration += duration
            
            # Calculate planned earned value as percentage
            planned_earned_value = (cumulative_duration / total_duration) * 100
            
            # Create enhanced task data
            enhanced_task = {
                'task_id': task_id,
                'name': task_name,
                'owner': owner,
                'duration': duration,
                'completion_date': completion_date,
                'earned_value': planned_earned_value,
                'cumulative_duration': cumulative_duration,
                'original_task': task  # Keep reference to original task data
            }
            enhanced_tasks.append(enhanced_task)
        
        # Step 6: Calculate summary statistics
        critical_tasks = [task for task in sorted_tasks if task.get('TF', float('inf')) == 0]
        
        summary_stats = {
            'total_tasks': len(sorted_tasks),
            'critical_path_tasks': len(critical_tasks),
            'average_task_duration': total_duration / len(sorted_tasks) if sorted_tasks else 0,
            'total_people_assigned': len(person_assignments),
            'project_completion_date': sorted_tasks[-1].get('Finish', 'N/A') if sorted_tasks else 'N/A'
        }
        
        return {
            'total_duration': total_duration,
            'sorted_tasks': enhanced_tasks,
            'task_owners': task_owners,
            'summary_stats': summary_stats
        }

    def get_project_dates(self, tasks: list) -> list:
        """
        Get all unique dates from task start and end dates, sorted chronologically.
        
        Args:
            tasks: List of scheduled task dictionaries (must have 'Start' and 'Finish' keys)
        
        Returns:
            List of unique dates sorted chronologically
        
        Raises:
            ValueError: If tasks list is empty
        """
        if not tasks:
            raise ValueError("Tasks list cannot be empty")
        
        dates = set()
        
        for task in tasks:
            start_date = task.get('Start')
            finish_date = task.get('Finish')
            
            if start_date and start_date != 'N/A':
                dates.add(start_date)
            
            if finish_date and finish_date != 'N/A':
                dates.add(finish_date)
        
        # Convert to sorted list
        return sorted(list(dates))

    def calculate_staffing_distribution(self, tasks: list, person_assignments: dict, task_resource_mapping: dict, core_team_resource_types: dict = None) -> dict:
        """
        Calculate staffing distribution over time by resource type.
        
        Args:
            tasks: List of scheduled task dictionaries (must have 'Start', 'Finish', 'Name' keys)
            person_assignments: Dictionary mapping person names to lists of assigned tasks
            task_resource_mapping: Dictionary mapping task names to required resource types
            core_team_resource_types: Optional dict mapping resource types to baseline counts (always available)
        
        Returns:
            Dictionary containing:
            - 'dates': List of unique project dates
            - 'resource_types': List of unique resource types
            - 'staffing_matrix': Dict mapping dates to resource type counts (includes core team)
            - 'person_to_resource': Dict mapping person names to resource types
            - 'core_team_counts': Dict mapping resource types to baseline counts
        
        Raises:
            ValueError: If inputs are invalid
        """
        if not tasks or not person_assignments or not task_resource_mapping:
            raise ValueError("All input parameters must be non-empty")
        
        # Initialize core team if not provided
        if core_team_resource_types is None:
            core_team_resource_types = {}
        
        # Get all unique dates
        project_dates = self.get_project_dates(tasks)
        
        # Get all unique resource types (from tasks + core team)
        task_resource_types = set(task_resource_mapping.values())
        core_resource_types = set(core_team_resource_types.keys())
        all_resource_types = task_resource_types.union(core_resource_types)
        resource_types = sorted(list(all_resource_types))
        
        # Create person to resource type mapping
        person_to_resource = {}
        for person, assigned_tasks in person_assignments.items():
            for task in assigned_tasks:
                task_name = task['Name']
                if task_name in task_resource_mapping:
                    person_to_resource[person] = task_resource_mapping[task_name]
                    break  # Each person should have consistent resource type
        
        # Initialize staffing matrix: {date: {resource_type: count}}
        staffing_matrix = {}
        for date in project_dates:
            staffing_matrix[date] = {resource_type: 0 for resource_type in resource_types}
        
        # Count active resources for each date
        for date in project_dates:
            active_persons = set()
            
            # Find all persons who have active tasks on this date
            for person, assigned_tasks in person_assignments.items():
                for task in assigned_tasks:
                    task_start = task.get('Start')
                    task_finish = task.get('Finish')
                    
                    # Check if task is active on this date
                    if (task_start and task_finish and 
                        task_start != 'N/A' and task_finish != 'N/A'):
                        if task_start <= date <= task_finish:
                            active_persons.add(person)
                            break  # Person is active, no need to check other tasks
            
            # Count actual people working by resource type
            actual_counts = {resource_type: 0 for resource_type in resource_types}
            for person in active_persons:
                if person in person_to_resource:
                    resource_type = person_to_resource[person]
                    actual_counts[resource_type] += 1
            
            # Use the maximum of core team requirement and actual people working
            for resource_type in resource_types:
                core_requirement = core_team_resource_types.get(resource_type, 0)
                actual_working = actual_counts[resource_type]
                staffing_matrix[date][resource_type] = max(core_requirement, actual_working)
        
        return {
            'dates': project_dates,
            'resource_types': resource_types,
            'staffing_matrix': staffing_matrix,
            'person_to_resource': person_to_resource,
            'core_team_counts': core_team_resource_types
        }

    def calculate_project_metrics(self, tasks: list, person_assignments: dict, task_resource_mapping: dict, core_team_resource_types: dict = None) -> dict:
        """
        Calculate comprehensive project metrics including cost, efficiency, and resource utilization.
        
        Args:
            tasks: List of scheduled task dictionaries (must have 'Start', 'Finish', 'Duration', 'Name' keys)
            person_assignments: Dictionary mapping person names to lists of assigned tasks
            task_resource_mapping: Dictionary mapping task names to required resource types
            core_team_resource_types: Optional dict mapping resource types to baseline counts
        
        Returns:
            Dictionary containing:
            - 'cost_man_months': Total actual effort in man-months
            - 'duration_months': Project duration in months
            - 'average_staffing': Average number of people working
            - 'estimated_effort_mm': Total estimated effort in man-months
            - 'efficiency_percent': Ratio of estimated to actual effort (%)
            - 'frontend_ratio_percent': Ratio of architect-only duration to total duration (%)
            - 'project_start_date': Project start date
            - 'project_end_date': Project end date
            - 'total_person_days': Total person-days of actual work
        
        Raises:
            ValueError: If inputs are invalid
        """
        if not tasks or not person_assignments or not task_resource_mapping:
            raise ValueError("All input parameters must be non-empty")
        
        if core_team_resource_types is None:
            core_team_resource_types = {}
        
        # Get staffing distribution data
        staffing_data = self.calculate_staffing_distribution(tasks, person_assignments, task_resource_mapping, core_team_resource_types)
        dates = staffing_data['dates']
        staffing_matrix = staffing_data['staffing_matrix']
        
        if not dates:
            raise ValueError("No valid project dates found")
        
        # Calculate project duration
        project_start_date = min(dates)
        project_end_date = max(dates)
        duration_days = (project_end_date - project_start_date).days + 1
        duration_months = duration_days / 30.44  # Average days per month
        
        # Calculate total actual effort (cost in man-months) by integrating staffing curve
        # This accounts for core team availability between milestone dates
        total_person_days = 0
        
        for i in range(len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i + 1]
            
            # Calculate number of days in this period
            period_days = (next_date - current_date).days
            
            # Get staffing level for this period (use current_date staffing)
            period_staffing = sum(staffing_matrix[current_date].values())
            
            # Add person-days for this period
            total_person_days += period_staffing * period_days
        
        cost_man_months = total_person_days / 30.44  # Convert person-days to man-months
        
        # Calculate average staffing
        average_staffing = total_person_days / duration_days if duration_days > 0 else 0
        
        # Calculate estimated effort (sum of all task durations)
        estimated_effort_days = sum(task.get('Duration', 0) for task in tasks)
        estimated_effort_mm = estimated_effort_days / 30.44  # Convert to man-months
        
        # Calculate efficiency (estimated / actual * 100)
        efficiency_percent = (estimated_effort_mm / cost_man_months * 100) if cost_man_months > 0 else 0
        
        # Calculate front-end ratio (architect-only duration vs total duration)
        # Find the period when only architect tasks are running
        architect_tasks = []
        non_architect_tasks = []
        
        for person, assigned_tasks in person_assignments.items():
            person_resource_type = None
            for task in assigned_tasks:
                task_name = task['Name']
                if task_name in task_resource_mapping:
                    person_resource_type = task_resource_mapping[task_name]
                    break
            
            for task in assigned_tasks:
                if person_resource_type == 'Architect':
                    architect_tasks.append(task)
                else:
                    non_architect_tasks.append(task)
        
        # Find the earliest non-architect task start date
        if non_architect_tasks:
            earliest_non_architect_start = min(task.get('Start') for task in non_architect_tasks if task.get('Start'))
        else:
            earliest_non_architect_start = project_end_date
        
        # Calculate frontend duration (from project start to first non-architect task)
        frontend_end_date = min(earliest_non_architect_start, project_end_date)
        frontend_duration_days = (frontend_end_date - project_start_date).days + 1
        
        # Ensure frontend duration is not negative and not longer than total project
        frontend_duration_days = max(0, min(frontend_duration_days, duration_days))
        
        frontend_ratio_percent = (frontend_duration_days / duration_days * 100) if duration_days > 0 else 0
        
        return {
            'cost_man_months': cost_man_months,
            'duration_months': duration_months,
            'average_staffing': average_staffing,
            'estimated_effort_mm': estimated_effort_mm,
            'efficiency_percent': efficiency_percent,
            'frontend_ratio_percent': frontend_ratio_percent,
            'project_start_date': project_start_date,
            'project_end_date': project_end_date,
            'total_person_days': total_person_days,
            'frontend_duration_days': frontend_duration_days,
            'total_project_days': len(dates)
        }

