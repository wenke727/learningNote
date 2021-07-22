# SUMO

----

## 安装

- error
    > The following packages have unmet dependencies:
    > Sumo : Depends: gdal-abi-2-2-3

  solution

    ```
    # step 1
    sudo apt-get install equivs

    # step2: create txt
    Section: misc
    Priority: optional
    Standards-Version: 3.9.2

    Package: gdal-abi-2-2-3
    Version: 2.2.3
    Depends: libgdal20
    Description: fake package for qgis which needs a gdal-abi-2-1-0

    # step 3 build gdal
    sudo equivs-build gdal_abi.txt
    sudo dpkg -i gdal-abi-2-2-3_2.2.3_all.deb

    # step 4: intall 
    sudo add-apt-repository ppa:sumo/stable
    sudo apt-get update
    sudo apt-get install sumo sumo-tools sumo-doc
    ```

- netconvert
  > netconvert: symbol lookup error: netconvert: undefined symbol: _ZN10OGRFeature16GetFieldAsStringEi

  solution:(ref: [GetFieldAsString() fails (aborts) on Ubuntu 8.10](https://trac.osgeo.org/gdal/ticket/2896))

    ```
    # copy files (`libgdal.so*`, `libogdi.so.3*`) from other sys
    sudo mv /home/pcl/Data/libgdal.so* /usr/lib/
    sudo mv /home/pcl/Data/libogdi.so.3* /usr/lib/
    ```

****

## [OpenStreetMap](https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html)

### Importing the Road Network

- cmd

```
/usr/bin/netconvert --osm-files hi-tech_park.osm.xml -o hi-tech_park.net.xml
```

- Recommended netconvert Options

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

- `vehicle-classes`: If "road" is given as parameter, only roads usable by road vehicles are extracted, if "passenger" is given, only those accessible by passenger vehicles.

- `type-file`: an additional output file with polygons of rivers and buildings as well as Points of Interest (POIs) will be generated
- `netconvert-options`: see Importing the Road Network
- `polyconvert-options`: see Importing additional Polygons (Buildings, Water, etc.)

### Further Notes

#### Junctions

In OpenStreetMap roads forming a single street and separated by, for example, a lawn or tram line, are represented by two edges that are parallel to each other. When crossing with another street, they form two junctions instead of one. To merge such junctions into a single junction, one can define which nodes to merge. See Networks/Building Networks from own XML-descriptions#Joining Nodes and netconvert documentation for usage details.

The netconvert option `--junctions.join` applies a heuristic to join these junction clusters automatically and is used by default when using the osmBuild.py script described above. However, some junction clusters are too complex for the heuristic and should be checked manually (as indicated by the warning messages). To manually specify joins for these junctions, see JoiningNodes Also, sometimes the heuristic wrongly joins some junctions. These can be excluded by giving them as a list to the option `--junctions.join-exclude`*.

When leaving junctions unjoined, there is a high risk of getting low throughput, jams and even deadlocks due to the short intermediate edges and the difficulty in computing proper traffic light plans for the junction clusters.

#### Traffic Lights

- Interpreting traffic light information in OSM
    netconvert prefers each intersection to be represented by a single node with a single traffic light controller. To achieve the former, see #Junctions. To achieve the latter some extra options are recommended. OSM often uses nodes ahead of an intersection to represent the position of traffic light signals. The actual intersection itself is then not marked as controlled. To interpret these structures the option --tls.guess-signals and --tls.guess-signals.dist <FLOAT> may be used. To cover the cases where this heuristic fails, the options below may be used to computed a joint tls plan for multiple nodes.

- Joining traffic lights
    OSM does not have the possibility to assign several nodes to a single traffic light. This means that near-by nodes, normally controlled by one traffic light system are controlled by two after the network is imported. It is obvious that traffic collapses in such areas if both traffic lights are not synchronized. Better representation of the reality can often be achieved by joining nearby junctions into a single junction. However, if the junctions should stay separate, it is possible to at least generate a joint controller by setting the option -`-tls.join`. For fine-tuning of joint traffic lights, the attribute tl can be customized for individual nodes.
    > testing

- Debugging missing traffic lights
- Overriding the traffic light information

****

## 解析osmWebWizard.py

1. netconvertOpts

```
../../bin/netconvert -t "D:\\Program Files\\Eclipse\\Sumo\\data\\typemap\\osmNetconvert.typ.xml" --geometry.remove --roundabouts.guess --ramps.guess -v --junctions.join --tls.guess-signals --tls.discard-simple --tls.join --output.original-names --junctions.corner-detail 5 --output.street-names --tls.default-type actuated --osm-files osm_bbox.osm.xml --keep-edges.by-vclass passenger -o osm.net.xml

../../bin/netconvert -c osm.netccfg
```

ubuntu version

```
export SUMO_HOME=/usr/share/sumo
/usr/share/sumo/bin/netconvert -t /usr/share/sumo/data/typemap/osmNetconvert.typ.xml --geometry.remove --roundabouts.guess --ramps.guess -v --junctions.join --tls.guess-signals --tls.discard-simple --tls.join --output.original-names --junctions.corner-detail 5 --output.street-names --tls.default-type actuated --osm-files osm_bbox.osm.xml --keep-edges.by-vclass passenger -o osm.net.xml

/usr/share/sumo/bin/netconvert -c osm.netccfg
```

2. polyconvertOpts

```
../../bin/polyconvert -v --osm.keep-full-type --type-file "D:\\Program Files\\Eclipse\\Sumo\\data\\typemap\\osmPolyconvert.typ.xml" --osm-files osm_bbox.osm.xml -n osm.net.xml -o osm.poly.xml --save-configuration osm.polycfg

../../bin/polyconvert -c osm.polycfg
```

3. Processing Cars

```
D:\Program Files\Eclipse\Sumo\bin\duarouter -n D:\Program Files\Eclipse\Sumo\tools\2021-03-29-16-23-14\osm.net.xml -r D:\Program Files\Eclipse\Sumo\tools\2021-03-29-16-23-14\osm.passenger.trips.xml --ignore-errors --begin 0 --end 3600 --no-step-log --no-warnings -o routes.rou.xml
D:\Program Files\Eclipse\Sumo\bin\duarouter -n D:\Program Files\Eclipse\Sumo\tools\2021-03-29-16-23-14\osm.net.xml -r D:\Program Files\Eclipse\Sumo\tools\2021-03-29-16-23-14\osm.passenger.trips.xml --ignore-errors --begin 0 --end 3600 --no-step-log --no-warnings -o D:\Program Files\Eclipse\Sumo\tools\2021-03-29-16-23-14\osm.passenger.trips.xml.tmp --write-trips

```

4. build.bat

```
python ../../tools/randomTrips.py -n osm.net.xml --fringe-factor 5 -p 33.068892 -o osm.passenger.trips.xml -e 3600 --vehicle-class passenger --vclass passenger --prefix veh --min-distance 300 --trip-attributes "departLane=\"best\"" --fringe-start-attributes "departSpeed=\"max\"" --allow-fringe.min-length 1000 --lanes --validate
```

5. Generating configuration file
   - osm.view.xml

   ```
   <viewsettings>
       <scheme name="real world"/>
       <delay value="20"/>
   </viewsettings>
   ```

    - Written configuration to 'osm.sumocfg'

    ```
    ..\..\bin\sumo -n osm.net.xml --gui-settings-file osm.view.xml --duration-log.statistics --device.rerouting.adaptation-interval 10 --device.rerouting.adaptation-steps 18 -v --no-step-log --save-configuration osm.sumocfg --ignore-route-errors -r osm.passenger.trips.xml -a osm.poly.xml
    ```

6. createBatch >> `run.bat`

```
sumo-gui -c osm.sumocfg
```

7. openSUMO

```
sumo-gui -c D:\\Program Files\\Eclipse\\Sumo\\tools\\2021-03-29-16-23-14\\osm.sumocfg
```

****

## ubuntu version

1. netconvertOpts
`/usr/share/sumo/bin/netconvert -c osm.netccfg` <- `Error: Could not access configuration 'osm.netccfg'.` [REF](https://sumo.dlr.de/docs/netconvert.html)

```
export SUMO_HOME=/usr/share/sumo
/usr/share/sumo/bin/netconvert -t /usr/share/sumo/data/typemap/osmNetconvert.typ.xml --geometry.remove --roundabouts.guess --ramps.guess -v --junctions.join --tls.guess-signals --tls.discard-simple --tls.join --output.original-names --junctions.corner-detail 5 --output.street-names --tls.default-type actuated --osm-files osm_bbox.osm.xml --keep-edges.by-vclass passenger -o osm.net.xml

/usr/share/sumo/bin/netconvert -c osm.netccfg
```

2. polyconvertOpts

```
/usr/share/sumo/bin/polyconvert -v --osm.keep-full-type --type-file /usr/share/sumo/data/typemap/osmPolyconvert.typ.xml --osm-files osm_bbox.osm.xml -n osm.net.xml -o osm.poly.xml --save-configuration osm.polycfg

/usr/share/sumo/bin/polyconvert -c osm.polycfg
```

3. build.bat

```
python /usr/share/sumo/tools/randomTrips.py -n osm.net.xml --fringe-factor 5 -p 33.068892 -o osm.passenger.trips.xml -e 3600 --vehicle-class passenger --vclass passenger --prefix veh --min-distance 300 --trip-attributes "departLane=\"best\"" --fringe-start-attributes "departSpeed=\"max\"" --allow-fringe.min-length 1000 --lanes --validate
```

4. Processing Cars

```
/usr/share/sumo/bin/duarouter -n ./osm.net.xml -r ./osm.passenger.trips.xml --ignore-errors --begin 0 --end 3600 --no-step-log --no-warnings -o routes.rou.xml
/usr/share/sumo/bin/duarouter -n ./osm.net.xml -r ./osm.passenger.trips.xml --ignore-errors --begin 0 --end 3600 --no-step-log --no-warnings -o ./osm.passenger.trips.xml.tmp --write-trips

```

5. Generating configuration file
   - osm.view.xml

   ```
   <viewsettings>
       <scheme name="real world"/>
       <delay value="20"/>
   </viewsettings>
   ```

    - Written configuration to 'osm.sumocfg'

    ```
    /usr/share/sumo/bin/sumo -n osm.net.xml --gui-settings-file osm.view.xml --duration-log.statistics --device.rerouting.adaptation-interval 10 --device.rerouting.adaptation-steps 18 -v --no-step-log --save-configuration osm.sumocfg --ignore-route-errors -r osm.passenger.trips.xml -a osm.poly.xml
    ```

6. createBatch >> `run.bat`

```
sumo-gui -c osm.sumocfg
```

7. openSUMO

```
sumo-gui -c D:\\Program Files\\Eclipse\\Sumo\\tools\\2021-03-29-16-23-14\\osm.sumocfg
```

****

# 路口拓宽操作

区域：科技中片区

- step 1

```

SUMO_HOME = "/usr/share/sumo"
osm_file = './osm_bbox.osm.xml'
name = 'osm'

pre_process = f' rm -r ./{name}; mkdir {name}; cp {osm_file} ./{name}/{osm_file}; cd ./{name}'

cmd = f"""
    {SUMO_HOME}/bin/netconvert  -t {SUMO_HOME}/data/typemap/osmNetconvert.typ.xml --geometry.remove --roundabouts.guess --ramps.guess -v \
    --junctions.join --tls.guess-signals --tls.discard-simple --tls.join --output.original-names --junctions.corner-detail 5 --output.street-names \
    --tls.default-type actuated --osm-files {osm_file} --keep-edges.by-vclass passenger -o {name}.net.xml
"""

# create node, edge files
cmd_tranfer0 = f"""{SUMO_HOME}/bin/netconvert --sumo-net-file {name}.net.xml --plain-output-prefix {name}; """
```

- step 2
  - osm.nod.xml
        add

        ```
        <node id="8349563238" x="2349.5" y="2131.77" type="priority"/>
        ```
  - osm.edge.xml
        modified

        ```
        <!-- 
        <edge id="208128052#0" from="1937711039" to="2184499023" name="科技中二路" priority="11" type="highway.secondary" numLanes="1" speed="27.78" shape="2355.35,1978.87 2355.07,1986.04 2349.49,2131.77 2349.18,2139.81 2344.45,2263.19 2343.93,2276.72" disallow="tram rail_urban rail rail_electric rail_fast ship">
        <lane index="0">
            <param key="origId" value="208128052"/>
        </lane>
        </edge> 
        -->

        <edge id="208128052#0" from="1937711039" to="8349563238" name="科技中二路" priority="11" type="highway.secondary" numLanes="1" speed="27.78" shape="2355.35,1978.87 2355.07,1986.04 2349.49,2131.77" disallow="tram rail_urban rail rail_electric rail_fast ship">
            <lane index="0">
                <param key="origId" value="208128052"/>
            </lane>
        </edge>

        <edge id="208128052#1" from="8349563238" to="2184499023" name="科技中二路" priority="11" type="highway.secondary" numLanes="2" speed="27.78" shape="2349.49,2131.77 2349.18,2139.81 2344.45,2263.19 2343.93,2276.72" disallow="tram rail_urban rail rail_electric rail_fast ship">
            <lane index="0">
                <param key="origId" value="208128052"/>
            </lane>
            <lane index="1">
                <param key="origId" value="208128052"/>
            </lane>
        </edge>
        ```
