
 $(document).ready(function(){
    var pawn_size = 45;




    function callbacks() {
        var selected_pawn = -1; 
         var player_color = 0;
         var pawn_destination = -1;

         $(".square").click(function(el) {
            square_loc = el["currentTarget"].classList[2];
            pawn_destination = square_loc;
            if (selected_pawn != -1 && pawn_destination != -1) {
                    move_uci = selected_pawn + pawn_destination;
                    send_move(move_uci);
                    selected_pawn = -1;
                    pawn_destination = -1;
                }
         });

         $("use").click(function(el) { 
            var a = el["currentTarget"]["attributes"][1].nodeValue;
            v = a.split("(")[1];
            w = v.split(")")[0];
            arr = v.split(",");

            x = parseInt(arr[0]);
            y = parseInt(arr[1]);
            id = Math.floor(x/pawn_size) + (Math.floor(y/pawn_size) * 8);
            href_attr = el["currentTarget"].getAttribute("xlink:href");
            if (href_attr.includes("white")) {
                x_1 = Math.floor(x/pawn_size);
                y_1 = (Math.floor(y/pawn_size));
                selected_pawn = String.fromCharCode(x_1+97) + "" + (8-(y_1)).toString();
            } else {
                x_1 = Math.floor(x/pawn_size);
                y_1 = (Math.floor(y/pawn_size));
                pawn_destination = String.fromCharCode(x_1+97) + "" + (8-(y_1)).toString();
                if (selected_pawn != -1 && pawn_destination != -1) {
                    move_uci = selected_pawn + pawn_destination;
                    send_move(move_uci);
                    selected_pawn = -1;
                    pawn_destination = -1;
                }
            }
            
        });
    

    }


    function send_move(move_uci) {
        $.getJSON( "/move/"+move_uci, function( data ) {
            if(data.hasOwnProperty("error")) {
                $("#result").text(data['error']);
            } else {
                $("#result").text("");
                $( "#board" ).load( "/board_svg", function() {
                    callbacks();
                });

            }
        });

    }

    $("#turnButton").click(function(el) {
        $.getJSON( "/next", function( data ) {
            $( "#board" ).load( "/board_svg", function() {
                callbacks();
            });
        });
    });

    
    $( "#board" ).load( "/board_svg", function() {
            callbacks();
    });

    

});