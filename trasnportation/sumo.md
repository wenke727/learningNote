## [OpenStreetMap](https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html)
****
### Importing the Road Network
* cmd
```
/usr/bin/netconvert --osm-files hi-tech_park.osm.xml -o hi-tech_park.net.xml
```

* Recommended netconvert Options
```
 --geometry.remove --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --tls.join --tls.default-type actuated
```

### Importing additional Polygons (Buildings, Water, etc.)

OSM-data not only contains the road network but also a wide range of additional polygons such as buildings and rivers. These polygons can be imported using polyconvert and then added to a sumo-gui-configuration.
```
/usr/bin/polyconvert --net-file hi-tech_park.net.xml --osm-files hi-tech_park.osm.xml --type-file /usr/share/sumo/data/typemap/osmPolyconvert.typ.xml -o hi-tech_park.poly.xml
```

The created polygon file berlin.poly.xml can then be added to a sumo-gui configuration:
```
 <configuration>
     <input>
         <net-file value="berlin.net.xml"/>
         <additional-files value="berlin.poly.xml"/>
     </input>
 </configuration>
```

### Import Scripts
The help script osmGet.py allows downloading a large area. The resulting file called "<PREFIX>.osm.xml" can then be imported using the script osmBuild.Py
```
osmGet.py --bbox <BOUNDING_BOX> --prefix <NAME>
osmBuild.py --osm-file <NAME>.osm.xml  [--vehicle-classes (all|road|passenger)] [--type-file <TYPEMAP_FILE>] [--netconvert-options <OPT1,OPT2,OPT3>] [--polyconvert-options <OPT1,OPT2,OPT3>]
```
* `vehicle-classes`: If "road" is given as parameter, only roads usable by road vehicles are extracted, if "passenger" is given, only those accessible by passenger vehicles.
* `type-file`: an additional output file with polygons of rivers and buildings as well as Points of Interest (POIs) will be generated
* `netconvert-options`: see Importing the Road Network
* `polyconvert-options`: see Importing additional Polygons (Buildings, Water, etc.)


### Further Notes
#### Junctions
In OpenStreetMap roads forming a single street and separated by, for example, a lawn or tram line, are represented by two edges that are parallel to each other. When crossing with another street, they form two junctions instead of one. To merge such junctions into a single junction, one can define which nodes to merge. See Networks/Building Networks from own XML-descriptions#Joining Nodes and netconvert documentation for usage details.

The netconvert option `--junctions.join` applies a heuristic to join these junction clusters automatically and is used by default when using the osmBuild.py script described above. However, some junction clusters are too complex for the heuristic and should be checked manually (as indicated by the warning messages). To manually specify joins for these junctions, see JoiningNodes Also, sometimes the heuristic wrongly joins some junctions. These can be excluded by giving them as a list to the option `--junctions.join-exclude`*.

When leaving junctions unjoined, there is a high risk of getting low throughput, jams and even deadlocks due to the short intermediate edges and the difficulty in computing proper traffic light plans for the junction clusters.

#### Traffic Lights
* Interpreting traffic light information in OSM
    netconvert prefers each intersection to be represented by a single node with a single traffic light controller. To achieve the former, see #Junctions. To achieve the latter some extra options are recommended. OSM often uses nodes ahead of an intersection to represent the position of traffic light signals. The actual intersection itself is then not marked as controlled. To interpret these structures the option --tls.guess-signals and --tls.guess-signals.dist <FLOAT> may be used. To cover the cases where this heuristic fails, the options below may be used to computed a joint tls plan for multiple nodes.

* Joining traffic lights
    OSM does not have the possibility to assign several nodes to a single traffic light. This means that near-by nodes, normally controlled by one traffic light system are controlled by two after the network is imported. It is obvious that traffic collapses in such areas if both traffic lights are not synchronized. Better representation of the reality can often be achieved by joining nearby junctions into a single junction. However, if the junctions should stay separate, it is possible to at least generate a joint controller by setting the option -`-tls.join`. For fine-tuning of joint traffic lights, the attribute tl can be customized for individual nodes.
    > testing

* Debugging missing traffic lights
* Overriding the traffic light information
