#!/bin/bash

set -e

DATADIR=/Users/matt/data/dragnet/cetr/tt/cleaneval/zh/

pushd $DATADIR

# rename HTML, copy over all .html files, delete the rest
mkdir HTML
mv original/*.html HTML
rm -r original

# rename/move the corrected files
mkdir Corrected
ls gold/*.txt | perl -pe 'm/\/([0-9]+)\.txt/; system "cp gold/$1.txt Corrected/$1.html.corrected.txt\n";'

# block_corrected
mkdir block_corrected

rm -r gold

# remove all files in HTML that are not in Corrected
# and vice versa
perl -e '
@html_files = glob "HTML/*";
@corrected_files = glob "Corrected/*";
%html = ();
foreach (@html_files) {
    $m/\/([0-9]+)\.html/;
    $html{$1} = 1;
}

%corrected = ();
foreach (@corrected_files) {
    $m/\/([0-9]+)\.html/;
    $corrected{$1} = 1;
}

# now check html.  if not in corrected then delete
while ( ($key, $value) = each(%html) ) {
    if (!defined $corrected{$key}) {
        system "rm HTML/$key.html";
    }
}

# vice versa
while ( ($key, $value) = each(%corrected) ) {
    if (!defined $html{$key}) {
        system "rm Corrected/$key.html.corrected.txt";
    }
}
'


# now need to massage the formatting of the files
# need to remove the first line
ls HTML/* Corrected/* | perl -pe '
$orig_file = $_;
open (F, $orig_file);
open (FOUT, ">t");
$k=0;
while (<F>) {
    if ($k > 0) {
        print FOUT $_;
    }
    $k++;
}
close(FOUT);
close(F);
system "mv t $orig_file";
'

rm -r stripped

# vips only exists for some of the data files
rm -r vips || true

popd


