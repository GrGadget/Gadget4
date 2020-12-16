#!/bin/bash

# $1 : source root directory
# $2 : build directory
# $3 : template file

DATE=$(git -C $1 log -n 1 2> /dev/null | head -n4 | grep "Date" | cut -d' ' -f4-)
COMMIT=$(git -C $1 log -n 1 2> /dev/null | head -n1 | grep "commit" | cut -d' ' -f2-)


if  [[ $DATE == "" ]]
    then
    DATE="unknown"
fi

if  [[ $COMMIT == "" ]]
    then
    COMMIT="unknown"
fi



if [ -f $BUILD_DIR/version.h ]
    then

    COMMIT2=$(grep "GIT_COMMIT" $2/version.h)

    if [[ $COMMIT2 == *$COMMIT* ]] #it's the same commit
    then
        exit
    fi
fi

cp $3 $2/version.h
sed -i.bu "s/_DATE_/$DATE/g" $2/version.h
sed -i.bu "s/_COMMIT_/$COMMIT/g" $2/version.h
rm $2/version.h.bu
